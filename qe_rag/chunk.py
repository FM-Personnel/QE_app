"""Découpage en chunks — revue § A1 + A2.

`app.py` :
  - A1 : `segment_text()` (découpe par titre) est calculée puis jamais utilisée ;
    seul `prepare_chunks_fixed()` (fenêtre 350/50) sert.
  - A2 : `prepare_chunks_fixed` appelle `detect_titles(w)` sur chaque MOT ; les
    regexs (`^Article\\s+\\d+`, …) ne matchent jamais un mot isolé -> le champ
    `section` du payload vaut « AUTRE » sur la quasi-totalité des chunks.

Ici :
  - `detect_titles()` s'applique à une LIGNE (comme prévu à l'origine) ;
  - `chunk_by_section()` (défaut) : 1 section = 1 chunk, recoupée si trop longue ;
  - `chunk_fixed()` : fenêtre fixe, mais la section est suivie ligne à ligne
    (séquence `(mot, section)`) donc la métadonnée reste juste.

Payload de sortie identique à celui attendu par `app.py::search_uploaded_documents`
et par les collections Qdrant existantes :
    {"id", "text", "metadata": {source, section, position, word_count, upload_date}}
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence

DEFAULT_CHUNK_SIZE = 350      # ~512 tokens
DEFAULT_OVERLAP = 50
DEFAULT_MAX_WORDS_SECTION = 240
MIN_CHUNK_WORDS = 50
MAX_CHUNKS_PER_DOC = 1500

# Marqueurs STRUCTURELS : ils ne peuvent pas apparaître au fil d'une phrase,
# donc ils valent titre quelle que soit la longueur de la ligne.
_TITLE_PATTERNS_STRUCTURELS = [
    re.compile(r"^#{1,6}\s+\S"),                              # titre markdown
    re.compile(r"^==== SECTION"),                             # marqueur de l'uploader app.py
]

# Motifs de PROSE : « Article 9 » est un titre, mais « article 53 de la loi
# n° 2018-727 du 10 août 2018 (ESSOC), article L. 313-1-2 du CASF… » est une
# phrase. Ces motifs ne valent donc QUE sur une ligne courte (cf. TITRE_LONGUEUR_MAX).
_TITLE_PATTERNS_PROSE = [
    re.compile(r"^Article\s+\d+", re.IGNORECASE),
    re.compile(r"^ANNEXE\s+\d+", re.IGNORECASE),
    re.compile(r"^PARTIE\s+\d+", re.IGNORECASE),
    re.compile(r"^(?:TITRE|Chapitre|Section)\s+\d+", re.IGNORECASE),
    # chiffres romains SUIVIS d'une lettre (pas d'un chiffre) : évite de prendre
    # « I. 2024 » ou une réf. d'article pour un titre.
    re.compile(r"^[IVXLCDM]{1,6}\.\s+[^\W\d_]", re.IGNORECASE),
]

# Garde-fou porté du notebook Colab (cellule 3), qui l'avait déjà résolu et
# documenté : « une phrase de prose demarrant par "Article 75 de la loi..." ou
# "L. 1432-1 et suivants..." ne doit pas etre prise pour un titre (sinon la
# section est coupee en plein milieu) ». `qe_rag` n'avait hérité que des motifs
# bruts, sans ce garde-fou : sur les fiches juridiques, une citation en tête de
# phrase ouvrait un chunk et la métadonnée `section` recevait un bout de
# citation. Mesuré : 350 chunks au lieu des 345 du notebook sur le lot de
# 25 fiches, l'écart concentré sur les 3 fiches les plus denses en références.
TITRE_LONGUEUR_MAX = 60

_TITLE_KEYWORDS = (
    "synthèse", "synthese", "conclusion", "introduction", "préambule", "preambule",
)


def detect_titles(line: str) -> bool:
    """True si la ligne ressemble à un titre / une tête de section.

    Ordre repris du notebook : les marqueurs structurels d'abord (ils valent
    titre même sur une ligne longue), PUIS le garde-fou de longueur, PUIS les
    motifs qui peuvent aussi ouvrir une phrase de prose.
    """
    line = (line or "").strip()
    if not line or len(line) > 200:
        return False
    if any(p.match(line) for p in _TITLE_PATTERNS_STRUCTURELS):
        return True
    if len(line) >= TITRE_LONGUEUR_MAX:
        return False
    if any(p.match(line) for p in _TITLE_PATTERNS_PROSE):
        return True
    low = line.lower()
    return any(low.startswith(k) for k in _TITLE_KEYWORDS)


def _clean_title(line: str) -> str:
    s = re.sub(r"^#{1,6}\s*", "", line.strip())
    s = s.replace("==== SECTION:", "").replace("====", "")
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip() or "AUTRE"


def _normalise_body(line: str) -> str:
    s = re.sub(r"^\s*[-*]\s+", "", line)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\bhttps?://\S+", "", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _noacc(s: str) -> str:
    """Minuscules sans accents — pour comparer des titres de section sans piéger
    sur « périmés » vs « perimes »."""
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def _footer_applicable(section: str, skip_footer_sections: Optional[Sequence[str]]) -> bool:
    """Le pied s'applique-t-il à cette section ?

    Certaines sections (annexes de service, listes de valeurs à ne plus citer)
    sont DENSES en sigles, numéros et dates : elles recouvrent lexicalement
    presque toute question du domaine et battent le chunk de contenu pertinent
    (c'est F9 retourné contre nous). Y empiler EN PLUS le pied aggrave le
    déséquilibre. On peut donc les exclure, par sous-chaîne, insensible aux
    accents et à la casse.

    Les titres concrètement exclus ne sont PAS écrits ici : ils viennent du
    manifeste (`scripts/rag_docs_manifest.json`, clé `skip_footer_sections`),
    source unique. Un test échoue si une de ces valeurs réapparaît dans `qe_rag`.
    """
    if not skip_footer_sections:
        return True
    titre = _noacc(section)
    return not any(_noacc(motif) in titre for motif in skip_footer_sections)


def _mk_chunk(text: str, source: str, section: str, position: int, footer: Optional[str],
              skip_footer_sections: Optional[Sequence[str]] = None) -> Dict:
    if footer and _footer_applicable(section, skip_footer_sections):
        text = f"{text}{footer}"
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{section}::{position}")),
        "text": text,
        "metadata": {
            "source": source,
            "section": section or "AUTRE",
            "position": position,
            "word_count": len(text.split()),
            "upload_date": datetime.now().isoformat(),
        },
    }


def _sections(lines: List[str]):
    """[(titre, [lignes de contenu])] — coupe sur les lignes-titres."""
    out, cur_title, buf = [], None, []
    for line in lines:
        if detect_titles(line):
            if buf:
                out.append((cur_title, buf))
            cur_title, buf = _clean_title(line), []
        else:
            buf.append(line)
    if buf:
        out.append((cur_title, buf))
    return [(t, b) for t, b in out if any(x.strip() for x in b)]


def chunk_by_section(
    lines: List[str],
    source: str,
    *,
    max_words: int = DEFAULT_MAX_WORDS_SECTION,
    min_words: int = MIN_CHUNK_WORDS,
    footer: Optional[str] = None,
    skip_footer_sections: Optional[Sequence[str]] = None,
    max_chunks: int = MAX_CHUNKS_PER_DOC,
) -> List[Dict]:
    """1 section = 1 chunk ; une section > `max_words` est recoupée.

    `skip_footer_sections` : titres (sous-chaînes) sur lesquels le pied n'est PAS
    ajouté — cf. `_footer_applicable`.
    """
    chunks: List[Dict] = []
    carry: List[str] = []  # mots d'une section trop courte, reportés sur la suivante

    def _extend_last(extra: List[str]) -> None:
        chunks[-1]["text"] += " " + " ".join(extra)
        chunks[-1]["metadata"]["word_count"] = len(chunks[-1]["text"].split())

    for title, body in _sections(lines):
        words = carry + " ".join(_normalise_body(x) for x in body).split()
        carry = []
        if len(words) < min_words:
            if chunks:
                _extend_last(words)          # rattache au chunk précédent
            else:
                carry = words                # rien avant : on reporte
            continue
        for k in range(0, len(words), max_words):
            piece = words[k : k + max_words]
            if len(piece) < min_words and chunks:
                _extend_last(piece)
                continue
            head = f"{title}. " if title and title != "AUTRE" else ""
            chunks.append(_mk_chunk((head + " ".join(piece)).strip(), source, title, len(chunks),
                                    footer, skip_footer_sections))
            if len(chunks) >= max_chunks:
                return chunks

    if carry:  # sections courtes en tête de document, jamais rattachées
        if chunks:
            _extend_last(carry)
        else:
            chunks.append(_mk_chunk(" ".join(carry), source, "AUTRE", 0, footer, skip_footer_sections))
    return chunks


def chunk_fixed(
    lines: List[str],
    source: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_words: int = MIN_CHUNK_WORDS,
    footer: Optional[str] = None,
    max_chunks: int = MAX_CHUNKS_PER_DOC,
) -> List[Dict]:
    """Fenêtre fixe avec recouvrement. La section est suivie ligne à ligne (fix A2)."""
    seq = []  # (mot, section)
    current = "AUTRE"
    for line in lines:
        if detect_titles(line):
            current = _clean_title(line)
            continue
        for w in _normalise_body(line).split():
            seq.append((w, current))

    chunks: List[Dict] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(seq) and len(chunks) < max_chunks:
        window = seq[start : start + chunk_size]
        if len(window) >= min_words:
            section = window[0][1]
            text = " ".join(w for w, _ in window)
            chunks.append(_mk_chunk(text, source, section, start, footer))
        start += step
    return chunks


def chunk(lines: List[str], source: str, *, strategy: str = "section", **kw) -> List[Dict]:
    if strategy == "section":
        return chunk_by_section(lines, source, **kw)
    if strategy == "fixed":
        return chunk_fixed(lines, source, **kw)
    raise ValueError(f"stratégie de chunking inconnue : {strategy!r} (section|fixed)")
