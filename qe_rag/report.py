"""Rapport de qualité d'une ingestion (revue § A6).

Le chemin d'upload de `app.py` renvoie `False` sur échec, sans dire à quelle
étape ni combien de caractères / chunks / vecteurs. Ici chaque étape est tracée
et le nombre de vecteurs réellement présents dans Qdrant est relu après l'upsert.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok " if self.ok else "ÉCHEC"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class IngestionReport:
    source: str
    collection: Optional[str] = None
    ok: bool = False
    dry_run: bool = False
    kind: Optional[str] = None
    chars_raw: int = 0
    chars_clean: int = 0
    n_chunks: int = 0
    n_words: int = 0
    n_vectors_upserted: int = 0
    n_vectors_in_collection: Optional[int] = None
    embedding_backend: Optional[str] = None
    steps: List[StepResult] = field(default_factory=list)
    error: Optional[str] = None
    sample_chunks: List[Dict[str, Any]] = field(default_factory=list)

    # -- construction --------------------------------------------------------
    def step(self, name: str, ok: bool, detail: str = "") -> StepResult:
        s = StepResult(name, ok, detail)
        self.steps.append(s)
        return s

    def fail(self, name: str, detail: str) -> "IngestionReport":
        self.step(name, False, detail)
        self.ok = False
        self.error = f"{name}: {detail}"
        return self

    # -- sérialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        lines = [
            f"document        : {self.source}",
            f"type            : {self.kind or '?'}",
            f"collection      : {self.collection or '(non créée)'}",
            f"backend embed   : {self.embedding_backend or '-'}",
            f"caractères      : {self.chars_raw:,} bruts -> {self.chars_clean:,} nettoyés",
            f"chunks          : {self.n_chunks} ({self.n_words:,} mots)",
        ]
        if not self.dry_run:
            in_coll = (
                self.n_vectors_in_collection
                if self.n_vectors_in_collection is not None
                else "?"
            )
            lines.append(
                f"vecteurs        : {self.n_vectors_upserted} envoyés, {in_coll} dans la collection"
            )
        lines.append("")
        lines += [str(s) for s in self.steps]
        lines.append("")
        lines.append(f"résultat        : {'OK' if self.ok else 'ÉCHEC'}")
        if self.error:
            lines.append(f"erreur          : {self.error}")
        if self.sample_chunks:
            lines.append("")
            lines.append("--- échantillon ---")
            for c in self.sample_chunks:
                meta = c.get("metadata", {})
                lines.append(
                    f"  pos {meta.get('position')} | section={meta.get('section')!r} | "
                    f"{meta.get('word_count')} mots"
                )
                lines.append("    " + c.get("text", "")[:240].replace("\n", " ") + " …")
        return "\n".join(lines)
