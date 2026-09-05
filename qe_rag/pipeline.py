"""Orchestration : extract -> clean -> chunk -> embed -> upsert.

Chaque étape est tracée dans un `IngestionReport` (revue § A6). Une étape qui
échoue n'interrompt pas par une exception : elle est consignée et le rapport
revient avec `ok=False` — l'appelant (CLI, futur chemin app) décide.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from . import chunk as _chunk
from . import clean as _clean
from . import extract as _extract
from . import qdrant_io as _q
from .embed import EmbeddingBackend
from .report import IngestionReport

MIN_TEXT_CHARS = 200
SAMPLE_CHUNKS = 3


def ingest_document(
    src: str,
    *,
    name: Optional[str] = None,
    kind: Optional[str] = None,
    collection: Optional[str] = None,
    backend: Optional[EmbeddingBackend] = None,
    qdrant_client=None,
    chunking: str = "section",
    chunk_size: int = _chunk.DEFAULT_CHUNK_SIZE,
    overlap: int = _chunk.DEFAULT_OVERLAP,
    max_words: int = _chunk.DEFAULT_MAX_WORDS_SECTION,
    min_words: Optional[int] = None,
    footer: Optional[str] = None,
    skip_footer_sections: Optional[Sequence[str]] = None,
    drop_summary: bool = True,
    recreate: bool = False,
    dry_run: bool = False,
    warn_duplicate: bool = True,
) -> IngestionReport:
    doc_name = name or _basename(src)
    rep = IngestionReport(source=doc_name, dry_run=dry_run)
    rep.embedding_backend = getattr(backend, "name", None)

    # 1. extraction ------------------------------------------------------
    try:
        raw, kind_eff = _extract.extract(src, kind)
        rep.kind = kind_eff
        rep.chars_raw = len(raw)
        if len(raw.strip()) < MIN_TEXT_CHARS:
            return rep.fail("extract", f"texte trop court ({len(raw.strip())} car.) — PDF image ? lien HTML ?")
        rep.step("extract", True, f"{len(raw):,} caractères ({kind_eff})")
    except Exception as exc:  # noqa: BLE001
        return rep.fail("extract", f"{type(exc).__name__}: {exc}")

    # 2. nettoyage -----------------------------------------------------
    try:
        lines = _clean.clean(raw, drop_summary=drop_summary)
        rep.chars_clean = sum(len(x) for x in lines)
        rep.step("clean", True, f"{len(lines)} lignes, {rep.chars_clean:,} caractères")
    except Exception as exc:  # noqa: BLE001
        return rep.fail("clean", f"{type(exc).__name__}: {exc}")

    # 3. chunking ----------------------------------------------------
    try:
        kw = dict(footer=footer)
        if chunking == "section":
            kw["max_words"] = max_words
            if min_words is not None:
                kw["min_words"] = min_words
            # exclusion du pied sur certaines sections : stratégie « section » seule
            # (le découpage à fenêtre fixe ne raisonne pas par titre de section)
            kw["skip_footer_sections"] = skip_footer_sections
        else:
            kw["chunk_size"] = chunk_size
            kw["overlap"] = overlap
        chunks = _chunk.chunk(lines, source=doc_name, strategy=chunking, **kw)
        rep.n_chunks = len(chunks)
        rep.n_words = sum(c["metadata"]["word_count"] for c in chunks)
        rep.sample_chunks = chunks[:SAMPLE_CHUNKS]
        n_sections = len({c["metadata"]["section"] for c in chunks})
        if not chunks:
            return rep.fail("chunk", "aucun chunk produit (texte sous le seuil minimal ?)")
        rep.step("chunk", True, f"{len(chunks)} chunks, {n_sections} sections, stratégie {chunking}")
    except Exception as exc:  # noqa: BLE001
        return rep.fail("chunk", f"{type(exc).__name__}: {exc}")

    if dry_run:
        rep.ok = True
        rep.step("dry-run", True, "arrêt avant embedding / upsert")
        return rep

    # 4. embeddings -----------------------------------------------
    if backend is None:
        return rep.fail("embed", "aucun backend d'embedding fourni")
    try:
        vectors = backend.encode([c["text"] for c in chunks])
        if len(vectors) != len(chunks):
            return rep.fail("embed", f"{len(vectors)} vecteurs pour {len(chunks)} chunks")
        dim = len(vectors[0])
        if dim != _q.VECTOR_SIZE:
            return rep.fail("embed", f"dimension {dim} != {_q.VECTOR_SIZE} (mauvais modèle ?)")
        rep.step("embed", True, f"{len(vectors)} vecteurs, dim {dim}, backend {rep.embedding_backend}")
    except Exception as exc:  # noqa: BLE001
        return rep.fail("embed", f"{type(exc).__name__}: {exc}")

    # 5. upsert Qdrant ------------------------------------------
    if qdrant_client is None:
        return rep.fail("upsert", "aucun client Qdrant fourni")
    try:
        if warn_duplicate and collection is None:
            dups = _q.find_existing_versions(qdrant_client, doc_name)
            if dups:
                rep.step("dedup", True, f"⚠️ version(s) déjà présente(s) : {', '.join(dups)} "
                                        f"(passer --collection pour remplacer)")
        coll = collection or _q.default_collection_name(doc_name)
        _q.ensure_collection(qdrant_client, coll, dim=len(vectors[0]), recreate=recreate)
        rep.collection = coll
        sent = _q.upsert_chunks(qdrant_client, coll, chunks, vectors)
        rep.n_vectors_upserted = sent
        rep.n_vectors_in_collection = _q.count(qdrant_client, coll)
        got = rep.n_vectors_in_collection
        ok_count = got is None or got >= sent
        rep.step("upsert", ok_count,
                 f"{sent} envoyés -> {got if got is not None else '?'} dans '{coll}'")

        # `recreate` = REMPLACER les versions précédentes du document, pas
        # seulement écraser un nom identique. Le nom portant un horodatage, la
        # suppression par nom exact ne trouvait jamais rien et les versions
        # s'empilaient. Fait ICI, après un upsert réussi et vérifié : la version
        # précédente reste servie tant que la nouvelle n'est pas en place.
        if recreate and ok_count:
            remplacees = _q.supprimer_versions_precedentes(qdrant_client, doc_name, sauf=coll)
            if remplacees:
                rep.step("remplacement", True,
                         f"{len(remplacees)} version(s) précédente(s) supprimée(s) : "
                         f"{', '.join(remplacees)}")
        rep.ok = ok_count
        if not ok_count:
            rep.error = f"comptage post-upsert {got} < {sent} envoyés"
    except Exception as exc:  # noqa: BLE001
        return rep.fail("upsert", f"{type(exc).__name__}: {exc}")

    return rep


def _basename(src: str) -> str:
    from pathlib import Path
    from urllib.parse import urlparse

    if src.startswith(("http://", "https://")):
        p = urlparse(src)
        return (Path(p.path).stem or p.netloc) or src
    return Path(src).stem
