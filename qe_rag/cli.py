"""CLI du pipeline d'ingestion.

    python -m qe_rag ingest <chemin|url> --name "Rapport IGAS MDPH (2024)"
    python -m qe_rag ingest doc.pdf --dry-run
    python -m qe_rag ingest https://exemple.gouv.fr/page --kind html \\
        --backend endpoint --endpoint-url https://<space>.hf.space
    python -m qe_rag embed "phrase de test" --backend endpoint --endpoint-url ...
    python -m qe_rag verify <collection>

Qdrant : QDRANT_URL / QDRANT_API_KEY depuis l'environnement ou un fichier .env.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _load_dotenv() -> None:
    """Charge .env sans dépendance (python-dotenv absent du venv de base)."""
    for cand in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
        if not cand.exists():
            continue
        for line in cand.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


# --- Manifeste des fiches curées : SOURCE UNIQUE ---------------------------
# `scripts/rag_docs_manifest.json` (maintenu par feat/fiches) porte la liste des
# fiches ET les paramètres de découpage du lot, dont `skip_footer_sections`.
# La CLI n'en garde AUCUNE copie : elle lit le manifeste. Une divergence entre
# les deux ne peut donc pas s'installer — c'est le motif du bug F12 (une même
# règle écrite à deux endroits qui finit par diverger).
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rag_docs_manifest.json"


def load_manifest(path: Optional[Path] = None) -> dict:
    """Charge le manifeste, ou `{}` s'il est absent (qe_rag reste utilisable
    hors de ce dépôt : sans manifeste, pas de liste d'exclusion par défaut)."""
    import json

    p = Path(path) if path else MANIFEST_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def default_skip_footer_sections() -> List[str]:
    """Liste d'exclusion du pied, lue dans le manifeste — jamais recopiée ici."""
    return list(load_manifest().get("skip_footer_sections") or [])


def _make_backend(args):
    from .embed import get_backend

    spec = args.backend
    kw = {}
    if "endpoint" in spec:
        kw["url"] = args.endpoint_url or os.getenv("EMBED_ENDPOINT_URL")
        if args.endpoint_token_env:
            kw["token"] = os.getenv(args.endpoint_token_env)
    if "local" in spec:
        if args.model:
            kw["model"] = args.model
        kw["device"] = args.device
    if spec.startswith("cache"):
        kw["cache_path"] = args.cache_path
    return get_backend(spec, **kw)


def _qdrant():
    from .qdrant_io import connect

    url = os.getenv("QDRANT_URL", "").strip()
    key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url.startswith("https://") or not key:
        sys.exit("❌ QDRANT_URL / QDRANT_API_KEY manquants (environnement ou .env)")
    return connect(url, key)


def cmd_ingest(args) -> int:
    from .pipeline import ingest_document

    backend = None if args.dry_run else _make_backend(args)
    client = None if args.dry_run else _qdrant()

    # Pied de chunk : texte ajouté à CHAQUE chunk (garde-fous « ne jamais citer
    # telle valeur périmée »). Lu depuis un fichier pour ne pas dépendre du
    # shell sur du texte long et accentué.
    footer = None
    if args.footer_file:
        footer = Path(args.footer_file).read_text(encoding="utf-8").strip()
        if footer and not footer.startswith(" "):
            footer = " " + footer

    rep = ingest_document(
        args.source,
        name=args.name,
        kind=args.kind,
        collection=args.collection,
        footer=footer,
        skip_footer_sections=args.skip_footer_section or None,
        min_words=args.min_words,
        backend=backend,
        qdrant_client=client,
        chunking=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_words=args.max_words,
        recreate=args.recreate,
        dry_run=args.dry_run,
        drop_summary=not args.keep_summary,
    )
    print(rep.to_json() if args.json else rep.summary())
    return 0 if rep.ok else 1


def cmd_embed(args) -> int:
    backend = _make_backend(args)
    vecs = backend.encode(list(args.text))
    print(f"backend={backend.name} dim={len(vecs[0])} n={len(vecs)}")
    for t, v in zip(args.text, vecs):
        print(f"  {t[:60]!r:64} -> [{v[0]:.4f}, {v[1]:.4f}, …]")
    return 0


def cmd_verify(args) -> int:
    client = _qdrant()
    from .qdrant_io import count

    n = count(client, args.collection)
    if n is None:
        print(f"❌ collection '{args.collection}' introuvable")
        return 1
    recs, _ = client.scroll(collection_name=args.collection, limit=3, with_payload=True)
    print(f"'{args.collection}' : {n} points")
    for r in recs:
        p = r.payload or {}
        print(f"  section={p.get('section')!r} | {p.get('word_count')} mots | source={p.get('source')!r}")
        print("    " + (p.get("text", "")[:200].replace("\n", " ")) + " …")
    return 0


def cmd_check(args) -> int:
    """Vérifie qu'un backend est réellement utilisable — sans charger de poids.

    Va plus loin qu'un test de présence : importe pour de vrai, donc attrape
    aussi les INCOMPATIBILITÉS entre paquets installés.
    """
    from .embed import verifier_backend, versions_backend

    if "local" in (args.backend or ""):
        v = versions_backend()
        print("versions installées : " + ", ".join(f"{k} {x}" for k, x in v.items()))

    probleme = verifier_backend(args.backend)
    if probleme:
        extras = " -r qe_rag/requirements-local.txt" if "local" in args.backend else ""
        print(f"❌ backend {args.backend!r} : {probleme}", file=sys.stderr)
        print(f"   pip install -r qe_rag/requirements.txt{extras}", file=sys.stderr)
        return 1
    print(f"✅ backend {args.backend!r} : utilisable")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qe_rag", description="Pipeline d'ingestion RAG QE Generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_backend_opts(p):
        p.add_argument("--backend", default="local",
                       help="local | endpoint | cache:local | cache:endpoint (défaut local)")
        p.add_argument("--endpoint-url", help="URL du service d'embedding (backend endpoint)")
        p.add_argument("--endpoint-token-env", default="EMBED_API_TOKEN",
                       help="nom de la variable d'env contenant le jeton bearer")
        p.add_argument("--model", help="chemin/repo du modèle local (backend local)")
        p.add_argument("--device", default="cpu")
        p.add_argument("--cache-path", default=".qe_rag_embed_cache.jsonl")

    pi = sub.add_parser("ingest", help="ingérer un document (PDF/DOCX/TXT/HTML ou URL)")
    pi.add_argument("source", help="chemin de fichier ou URL")
    pi.add_argument("--name", help="nom lisible du document (défaut : nom de fichier)")
    pi.add_argument("--kind", choices=["pdf", "docx", "txt", "html"], help="forcer le type")
    pi.add_argument("--collection", help="nom exact de collection (défaut : <slug>__<timestamp>)")
    pi.add_argument("--chunking", choices=["section", "fixed"], default="section")
    pi.add_argument("--chunk-size", type=int, default=350)
    pi.add_argument("--overlap", type=int, default=50)
    pi.add_argument("--max-words", type=int, default=240, help="taille max d'un chunk de section")
    pi.add_argument("--recreate", action="store_true", help="supprimer la collection si elle existe")
    pi.add_argument("--keep-summary", action="store_true", help="ne pas retirer le sommaire")
    pi.add_argument("--footer-file",
                    help="fichier texte ajouté à la fin de CHAQUE chunk (garde-fous chiffres périmés)")
    pi.add_argument("--skip-footer-section", action="append", metavar="MOTIF",
                    default=default_skip_footer_sections(),
                    help="titre de section (sous-chaîne, sans accent) sur lequel le pied n'est PAS "
                         "ajouté ; répétable. Par défaut : la liste du manifeste "
                         "(scripts/rag_docs_manifest.json), jamais une copie — ces sections de "
                         "service sont denses en sigles et numéros, elles battent le chunk de "
                         "contenu si on empile le pied dessus (F9 retourné contre nous).")
    pi.add_argument("--min-words", type=int, default=None,
                    help="taille minimale d'un chunk de section (défaut qe_rag : 50). Le notebook "
                         "Colab utilise 20 : passer 20 pour reproduire son découpage.")
    pi.add_argument("--dry-run", action="store_true", help="extract + clean + chunk seulement")
    pi.add_argument("--json", action="store_true", help="rapport en JSON")
    add_backend_opts(pi)
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("embed", help="encoder du texte (test d'un backend)")
    pe.add_argument("text", nargs="+")
    add_backend_opts(pe)
    pe.set_defaults(func=cmd_embed)

    pc = sub.add_parser("check", help="vérifier les dépendances d'un backend (sans rien charger)")
    pc.add_argument("--backend", default="local",
                    help="local | endpoint | cache:local | cache:endpoint (défaut local)")
    pc.set_defaults(func=cmd_check)

    pv = sub.add_parser("verify", help="compter et échantillonner une collection Qdrant")
    pv.add_argument("collection")
    pv.set_defaults(func=cmd_verify)

    return ap


def main(argv=None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
