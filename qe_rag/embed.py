"""Backends d'embedding pluggables — revue § C2.

- `local`    : sentence-transformers en local (Colab, ou PC ≥ 8 Go de RAM) ;
- `endpoint` : service HTTP (le Space `services/embedding/`, contrat POST /embed) ;
- `cache`    : enrobe un autre backend + un cache disque (JSONL) — pour l'éval,
               qui rejoue les mêmes textes.

Tous exposent `encode(texts: list[str]) -> list[list[float]]` et l'attribut `dim`.
Les vecteurs ne sont PAS normalisés (cohérence avec `app.py` et les collections
Qdrant existantes, distance Cosine).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

HUB_MODEL = "Whisler/camembert_finetuned_progressive"
LOCAL_MODEL_DIR = "./models/camembert_finetuned_progressive"
EXPECTED_DIM = 1024


class EmbeddingBackend(ABC):
    name: str = "abstract"
    dim: Optional[int] = None

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


# --------------------------------------------------------------------------- #
class LocalBackend(EmbeddingBackend):
    name = "local"

    def __init__(self, model: Optional[str] = None, device: str = "cpu", batch_size: int = 16):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "backend 'local' : sentence-transformers absent "
                "(`pip install sentence-transformers`)"
            ) from exc
        path = model or (LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HUB_MODEL)
        self._model = SentenceTransformer(path, device=device)
        self.batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=False, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vecs]


# --------------------------------------------------------------------------- #
class EndpointBackend(EmbeddingBackend):
    name = "endpoint"

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        batch_size: int = 64,
        timeout: int = 90,
        retries: int = 3,
    ):
        try:
            import requests  # noqa: F401
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("backend 'endpoint' : `pip install requests`") from exc
        self.url = url.rstrip("/")
        self.embed_url = self.url if self.url.endswith("/embed") else f"{self.url}/embed"
        self.token = token
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self.dim = None

    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.token:
            h["authorization"] = f"Bearer {self.token}"
        return h

    def _post_batch(self, batch: List[str]) -> List[List[float]]:
        import requests

        payload = json.dumps({"inputs": batch, "normalize": False})
        last_err = None
        for attempt in range(self.retries):
            try:
                r = requests.post(
                    self.embed_url, data=payload, headers=self._headers(), timeout=self.timeout
                )
                r.raise_for_status()
                data = r.json()
                vecs = data.get("embeddings")
                if not isinstance(vecs, list) or len(vecs) != len(batch):
                    raise ValueError(f"réponse inattendue : {str(data)[:200]}")
                if self.dim is None and vecs:
                    self.dim = len(vecs[0])
                return vecs
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"endpoint {self.embed_url} : {last_err}")

    def encode(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post_batch(texts[i : i + self.batch_size]))
        return out


# --------------------------------------------------------------------------- #
class CacheBackend(EmbeddingBackend):
    name = "cache"

    def __init__(self, inner: EmbeddingBackend, path: str):
        self.inner = inner
        self.path = Path(path)
        self._cache: dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._cache[rec["k"]] = rec["v"]
        self.dim = inner.dim
        if self._cache and self.dim is None:
            self.dim = len(next(iter(self._cache.values())))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def encode(self, texts: List[str]) -> List[List[float]]:
        keys = [self._key(t) for t in texts]
        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]
        if missing:
            self.misses += len(missing)
            fresh = self.inner.encode([t for _, t in missing])
            self.dim = self.inner.dim or (len(fresh[0]) if fresh else self.dim)
            with self.path.open("a", encoding="utf-8") as fh:
                for (idx, _), vec in zip(missing, fresh):
                    self._cache[keys[idx]] = vec
                    fh.write(json.dumps({"k": keys[idx], "v": vec}) + "\n")
        self.hits += len(texts) - len(missing)
        return [self._cache[k] for k in keys]


# --------------------------------------------------------------------------- #
# Dépendances par backend — déclarées ICI, à côté des backends eux-mêmes, pour
# que le workflow d'ingestion n'ait pas à en garder une copie (c'est une
# duplication de ce genre qui a produit F12). `pip install -r` correspondant
# indiqué avec, pour que le message d'erreur soit actionnable.
_DEPENDANCES_BACKEND = {
    "local": [("sentence_transformers", "sentence-transformers"), ("torch", "torch")],
    "endpoint": [("requests", "requests")],
}


def versions_backend() -> dict:
    """Versions réellement résolues des paquets du chemin « local ».

    Sert au diagnostic : une incompatibilité de versions ne se voit pas dans un
    « paquet absent », elle se voit dans le COUPLE installé.
    """
    versions = {}
    for module in ("torch", "transformers", "sentence_transformers"):
        try:
            versions[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            versions[module] = f"(indisponible : {type(exc).__name__})"
    return versions


def verifier_backend(spec: str) -> Optional[str]:
    """Vérifie qu'un backend est réellement UTILISABLE. `None` si tout va bien.

    `dependances_manquantes()` ne fait que chercher les modules (`find_spec`),
    sans les exécuter : elle attrape « paquet absent », pas « paquets présents
    mais incompatibles entre eux ». C'est précisément ce qui est passé au
    travers lors du 2ᵉ lancement du workflow — `sentence-transformers` 6.0.1
    et `transformers` 5.16.1 installés, mais cassant à l'import contre un
    `torch` 2.2.2 épinglé trop bas.

    On IMPORTE donc pour de vrai, et on exerce torch, sans jamais télécharger
    ni charger de poids de modèle : c'est l'import qui casse dans ce cas, pas
    le chargement.
    """
    manquants = dependances_manquantes(spec)
    if manquants:
        return f"dépendance(s) absente(s) : {', '.join(manquants)}"

    if "local" not in (spec or ""):
        return None

    try:
        import torch  # noqa: F401

        torch.zeros(1)  # le runtime torch fonctionne, pas seulement l'import
    except Exception as exc:  # noqa: BLE001
        return f"torch inutilisable : {type(exc).__name__}: {exc}"

    try:
        # L'import qui casse réellement en cas de couple torch/transformers
        # incompatible (il tire `transformers.integrations`).
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        v = versions_backend()
        return (f"sentence-transformers inutilisable : {type(exc).__name__}: {exc} "
                f"— versions installées : torch {v['torch']}, "
                f"transformers {v['transformers']}, "
                f"sentence-transformers {v['sentence_transformers']}. "
                f"C'est une INCOMPATIBILITÉ de versions, pas un paquet manquant.")
    return None


def dependances_manquantes(spec: str) -> List[str]:
    """Modules requis par un backend et absents de l'environnement.

    Vérifie par IMPORT, sans charger de modèle ni ouvrir de connexion : c'est
    fait pour échouer en quelques secondes au début d'un run, plutôt qu'après
    une longue préparation — voire, pour l'ingestion, après avoir touché à
    Qdrant. Renvoie les noms de PAQUET (pip), pas les noms de module.
    """
    import importlib.util

    famille = "local" if "local" in (spec or "") else "endpoint" if "endpoint" in (spec or "") else None
    if famille is None:
        return []
    manquants = []
    for module, paquet in _DEPENDANCES_BACKEND[famille]:
        if importlib.util.find_spec(module) is None:
            manquants.append(paquet)
    return manquants


def get_backend(spec: str = "local", **kw) -> EmbeddingBackend:
    """Fabrique.

    spec :
      "local"                          -> LocalBackend(model=?, device=?)
      "endpoint"                       -> EndpointBackend(url=, token=?)
      "cache:local" / "cache:endpoint" -> CacheBackend enrobant le backend nommé
                                          (kw `cache_path`, défaut .qe_rag_embed_cache.jsonl)
    """
    if spec.startswith("cache"):
        inner_spec = spec.split(":", 1)[1] if ":" in spec else "local"
        cache_path = kw.pop("cache_path", ".qe_rag_embed_cache.jsonl")
        return CacheBackend(get_backend(inner_spec, **kw), cache_path)
    if spec == "local":
        return LocalBackend(
            model=kw.get("model"), device=kw.get("device", "cpu"),
            batch_size=kw.get("batch_size", 16),
        )
    if spec == "endpoint":
        url = kw.get("url") or os.getenv("EMBED_ENDPOINT_URL")
        if not url:
            raise ValueError("backend 'endpoint' : fournir url= ou EMBED_ENDPOINT_URL")
        return EndpointBackend(
            url=url, token=kw.get("token"),
            batch_size=kw.get("batch_size", 64), timeout=kw.get("timeout", 90),
        )
    raise ValueError(f"backend inconnu : {spec!r} (local|endpoint|cache:*)")
