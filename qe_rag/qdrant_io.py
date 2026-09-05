"""Écriture dans Qdrant — création de collection, upsert par lots, relecture du
compte réel (revue § A6 : `app.py` a le comptage post-indexation commenté).

Nom de collection : `<slug>__<timestamp>` par défaut (compat avec l'existant et
avec `search_uploaded_documents`, qui reconnaît le préfixe de slug
`Fiche_de_reference`). Un nom explicite + `recreate=True` permet une ré-ingestion
idempotente (revue § B3, traité a minima).
"""
from __future__ import annotations

import re
import time
from typing import Iterable, List, Optional

VECTOR_SIZE = 1024
PROTECTED = {
    "QuestionParlementaire",
    "Code de la sécurité sociale",
    "Code du travail",
    "CASF",
    "Code de la santé publique",
    "CodesJuridiques",
}


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s().\-]", "", name, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s)[:90]


def default_collection_name(doc_name: str, ts: Optional[int] = None) -> str:
    return f"{slugify(doc_name)}__{ts or int(time.time())}"


def connect(url: str, api_key: str, timeout: int = 60):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)


def find_existing_versions(client, doc_name: str) -> List[str]:
    """Collections dont le slug (avant `__`) correspond au document — pour repérer les doublons."""
    slug = slugify(doc_name).lower()
    out = []
    for c in client.get_collections().collections:
        base = c.name.split("__")[0].lower()
        if base == slug:
            out.append(c.name)
    return out


def supprimer_versions_precedentes(client, doc_name: str, *, sauf: Optional[str] = None) -> List[str]:
    """Supprime toutes les collections `<slug>__*` du document, SAUF `sauf`.

    Pourquoi ce n'est pas `ensure_collection(recreate=True)` : le nom d'une
    collection porte un HORODATAGE, donc un nouveau run produit toujours un nom
    neuf. Supprimer « la collection de même nom » ne trouve alors jamais rien,
    et chaque ingestion EMPILE une version de plus. C'est ce qui a produit 50
    collections (703 points) là où on en attendait 25 (347) : les 25 fiches
    existaient en double, et le retrieval voyait les deux.

    À appeler APRÈS un upsert réussi, pas avant : l'ancienne version reste en
    place tant que la nouvelle n'est pas écrite et vérifiée. On remplace, on ne
    supprime-puis-écrit pas.
    """
    supprimees = []
    for nom in find_existing_versions(client, doc_name):
        if nom == sauf or nom in PROTECTED:
            continue
        client.delete_collection(nom)
        supprimees.append(nom)
    return supprimees


def compter_versions(client, doc_name: str) -> List[str]:
    """Collections `<slug>__*` actuellement présentes — pour contrôler l'état
    RÉEL de la base après ingestion, et pas seulement ce qu'on a écrit."""
    return find_existing_versions(client, doc_name)


def ensure_collection(client, name: str, *, dim: int = VECTOR_SIZE, recreate: bool = False) -> str:
    from qdrant_client.http import models as qm

    if name in PROTECTED:
        raise ValueError(f"'{name}' est une collection protégée")
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    return name


def upsert_chunks(client, name: str, chunks: List[dict], vectors: List[List[float]],
                  *, batch_size: int = 64, retries: int = 3) -> int:
    """Upsert par lots avec réessais. Renvoie le nombre de points envoyés."""
    from qdrant_client.http import models as qm

    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks vs {len(vectors)} vecteurs")

    points = [
        qm.PointStruct(id=c["id"], vector=v, payload={"text": c["text"], **c["metadata"]})
        for c, v in zip(chunks, vectors)
    ]
    sent = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        last_err = None
        for attempt in range(retries):
            try:
                client.upsert(collection_name=name, points=batch, wait=True)
                sent += len(batch)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"upsert lot {i // batch_size + 1} : {last_err}")
    return sent


def count(client, name: str) -> Optional[int]:
    try:
        return client.count(name).count
    except Exception:  # noqa: BLE001
        return None
