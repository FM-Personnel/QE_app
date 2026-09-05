"""Nettoyage du texte extrait — revue § A3.

`app.py::clean_text` aplatit TOUS les retours à la ligne (`re.sub(r'\\s+', ' ')`
sur le texte entier) : la structure (titres, sections) est perdue avant le
chunking, ce qui casse la découpe par section (§ A1/A2).

Ici on garde les lignes : on filtre les lignes-parasites, on normalise CHAQUE
ligne indépendamment, et on rend un texte encore découpé en lignes. Le chunker
(`chunk.py`) s'appuie sur ces frontières de ligne pour repérer les titres.
"""
from __future__ import annotations

import re
from typing import List

_JUNK_LINE_PATTERNS = [
    re.compile(r"^(?:Page\s*)?\d{1,4}\s*$", re.IGNORECASE),          # numéros de page isolés
    re.compile(
        r"^(?:PLFSS\s*\d{4}\s*-\s*Annexe\s*\d+"
        r"|ANNEXE\s*DÉPENSES\s*DE\s*LA\s*BRANCHE\s*.*"
        r"|securite-sociale\.fr"
        r"|G[ée]n[ée]ration\s*X-Book)$",
        re.IGNORECASE,
    ),
    re.compile(r"^[•○◘\-—~=_.]{2,}$"),                                 # lignes de séparation
]

# Un sommaire = une ligne qui n'est QUE « SOMMAIRE » / « TABLE DES MATIÈRES »,
# suivie de lignes jusqu'à la première vraie tête de section. Si aucune tête de
# section ne suit, on ne touche à rien (mieux vaut garder le sommaire que vider
# le document).
_SUMMARY_RE = re.compile(
    r"(?im)^[ \t]*(?:SOMMAIRE|TABLE\s+DES\s+MATI[ÈE]RES)[ \t]*$"
    r".*?"
    r"(?=^[ \t]*(?:PARTIE\s+\d+|ANNEXE\s+\d+|Article\s+\d+|#{1,6}\s|==== SECTION))",
    re.DOTALL,
)


def normalise_line(s: str) -> str:
    """Normalise une ligne : espaces, mots coupés par tiret, URLs, artefacts."""
    s = re.sub(r"^#+\s*", "", s)                 # marqueurs de titre markdown
    s = re.sub(r"^\s*[-*]\s+", "", s)            # puces
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)       # gras markdown
    s = re.sub(r"\bhttps?://\S+", "", s)
    s = re.sub(r"[•○◘]+|[-=~]{5,}", "", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)      # mot coupé par un retour à la ligne : « assu- rance »
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_lines(text: str) -> List[str]:
    """Filtre les lignes-parasites. Ne normalise pas encore (le chunker veut les titres bruts)."""
    out: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(p.match(line) for p in _JUNK_LINE_PATTERNS):
            continue
        out.append(line)
    return out


def remove_summary(text: str) -> str:
    """Retire un sommaire / une table des matières en tête de document."""
    cleaned = _SUMMARY_RE.sub("", text, count=1)
    return cleaned.strip() if cleaned.strip() else text


def clean(text: str, drop_summary: bool = True) -> List[str]:
    """Point d'entrée : texte brut -> liste de lignes nettoyées."""
    if drop_summary:
        text = remove_summary(text)
    return clean_lines(text)
