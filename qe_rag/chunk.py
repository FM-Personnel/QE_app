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


# --- Découpage aux frontières de phrase — ACTIF depuis le 07/09/2026 --------
#
# `chunk_by_section` coupe une section longue au NOMBRE DE MOTS, donc en pleine
# phrase, et le pied de page est ensuite collé à chaque morceau : mesuré sur les
# 659 chunks porteurs de la base, 106 (16 %) ont un corps interrompu au milieu
# d'une phrase juste avant leur pied (cas #11943). Le correctif applicatif du
# 06/09 (`isoler_pied_du_corps`) empêche le pied de se lire comme la suite de la
# phrase, mais ne recolle pas la phrase : seule une réingestion le ferait.
#
# ACTIVÉ sur arbitrage de l'utilisateur (07/09), avec la réingestion des 35
# fiches. À False, `_bornes_decoupe` rend EXACTEMENT les bornes de
# `range(0, len(words), max_words)` — le comportement d'avant, au mot près ; le
# réglage reste donc réversible sans rien réécrire.
#
# ⚠️ Le notebook Colab (`scripts/build_curated_notebook.py`, cellule 3) porte le
# MÊME découpage. `tests/qe_rag/test_parite_notebook.py` échoue si les deux
# divergent, et il injecte `MARGE_PHRASE` depuis ce fichier — ne pas changer la
# marge ici sans régénérer le notebook.
DECOUPAGE_AUX_PHRASES = True

# Marge de recul : on accepte de raccourcir un morceau d'au plus cette fraction
# de `max_words` pour tomber sur une fin de phrase ; au-delà, coupure au mot.
# 0,50 mesuré meilleur que 0,25 sur LES DEUX axes à la fois — 7 coupures avec
# pied au lieu de 33, et un écart-type de tailles PLUS FAIBLE (56,7 contre
# 59,7). Il n'y avait pas d'arbitrage propreté / régularité à faire ; le seul
# coût est +11 chunks.
MARGE_PHRASE = 0.50

# Un mot termine une phrase s'il finit par une ponctuation forte, éventuellement
# suivie d'un guillemet ou d'une parenthèse fermante.
_FIN_PHRASE = re.compile(r"[.!?…][\"'»)\]]*$")

# ⚠️ Faux positifs à écarter : en français administratif, « art. », « n° » et les
# abréviations de civilité finissent par un point sans terminer la phrase. Une
# coupure après elles serait exactement le défaut qu'on corrige.
#
# ⚠️⚠️ LACUNE ASSUMÉE, NE PAS « CORRIGER » SANS LIRE CECI. La liste contient
# `II.` à `VI.` mais PAS `I.`, `VII.`, `VIII.`, `IX.`, `X.`, ni l'ordinal
# `Ier.` — or ce sont les alinéas du droit français. Ce n'est pas un oubli :
#
#   · mesuré sur 12 000 chunks de codes (35 823 tokens acceptés comme fins de
#     phrase), ces alinéas pèsent 0,25 % des acceptations, et les énumérations
#     `1°.` `2°.` 0,21 % ;
#   · surtout, leur EFFET est bénin : une coupure avant un alinéa ou avant un
#     item d'énumération tombe sur une frontière structurelle saine. Ce n'est
#     PAS le défaut qu'on corrige, qui est la coupure au milieu d'une phrase.
#
# Compléter la liste serait du zèle, et le zèle sur un détecteur est ce qui
# l'abîme : chaque entrée ajoutée est une coupure propre qu'on s'interdit.
# Détail et mesure : `reference/note_pied_garanti_et_codes.md`.
_ABREVIATIONS = {
    "art.", "arts.", "cf.", "etc.", "ex.", "p.", "pp.", "al.", "chap.",
    "M.", "MM.", "Mme.", "Mmes.", "Dr.", "Pr.", "no.", "n°.", "s.",
    "L.", "R.", "D.", "II.", "III.", "IV.", "V.", "VI.",
}


def _fin_de_phrase(mot: str) -> bool:
    """Ce mot termine-t-il une phrase ?"""
    if mot in _ABREVIATIONS:
        return False
    # ⚠️ L'EXCEPTION VIENT AVANT L'EXCLUSION, et l'ordre est le correctif.
    # Un token de QUATRE chiffres suivi d'un point termine une phrase : « … en
    # vigueur le 1er janvier 2025. » L'exclusion générale ci-dessous, écrite
    # pour les puces d'énumération, l'avalait aussi — et c'est le cas le plus
    # fréquent du corpus. Sûreté vérifiée sur les tokens réels des 35 fiches :
    # 334 occurrences, 28 valeurs, dont 3 hors plage de millésime (`3977.`,
    # `3133.`, `9999.`) — toutes de vraies fins de phrase, relues une à une.
    # Aucun faux positif. La règle porte donc sur les 4 chiffres, pas sur la
    # plage 1900-2099.
    #
    # Étroite à dessein : « 2025) » reste exclu, une parenthèse fermante après
    # un nombre est une énumération, pas une fin de phrase.
    if re.fullmatch(r"\d{4}\.", mot):
        return True
    # « 12. » ou « 1) » : un nombre suivi d'un point ou d'une parenthèse est
    # une énumération ou un millésime tronqué, pas une fin de phrase.
    if re.fullmatch(r"\d+[.)]", mot):
        return False
    # une initiale seule : « J. » dans « J. Dupont »
    if re.fullmatch(r"[A-ZÉÈÀÇ]\.", mot):
        return False
    return bool(_FIN_PHRASE.search(mot))


def _bornes_decoupe(words: List[str], max_words: int) -> List[tuple]:
    """Bornes (début, fin) des morceaux d'une section.

    Réglage inerte : à `DECOUPAGE_AUX_PHRASES = False`, rend les mêmes bornes
    que `range(0, len(words), max_words)`.
    """
    if not DECOUPAGE_AUX_PHRASES:
        return [(k, min(k + max_words, len(words)))
                for k in range(0, len(words), max_words)]

    marge = max(1, int(max_words * MARGE_PHRASE))
    bornes, debut = [], 0
    while debut < len(words):
        fin = min(debut + max_words, len(words))
        if fin < len(words):
            # on recule jusqu'à la dernière fin de phrase dans la marge
            recul = fin
            while recul > fin - marge and recul > debut + 1:
                if _fin_de_phrase(words[recul - 1]):
                    break
                recul -= 1
            if recul > debut + 1 and _fin_de_phrase(words[recul - 1]):
                fin = recul          # coupure propre
            # sinon : aucune frontière dans la marge, on garde la coupure au mot
        bornes.append((debut, fin))
        debut = fin
    return bornes


# --- Pose du pied à la FIN du découpage — INERTE par défaut -----------------
#
# `_extend_last` rattache une section trop courte au chunk précédent **après**
# que celui-ci a reçu son pied : du contenu se retrouve donc APRÈS le pied.
# Mesuré : 30 chunks sur 836. Poser le pied une fois le découpage terminé les
# supprime par construction — la fusion a déjà eu lieu quand le pied arrive.
#
# ACTIVÉ le 07/09 dans le même lot. À False, le pied est posé à la création,
# comportement d'avant.
PIED_POSE_A_LA_FIN = True


def _poser_pieds(chunks: List[Dict], footer: Optional[str],
                 skip_footer_sections: Optional[Sequence[str]]) -> List[Dict]:
    """Appose le pied à chaque chunk, une fois le découpage terminé."""
    if not footer:
        return chunks
    for c in chunks:
        # ⚠️ TROISIÈME CONSÉQUENCE de l'étiquette périmée, trouvée le 09/09 en
        # cherchant la suivante — et c'est la plus grave des trois.
        # Cette fonction décidait du pied D'APRÈS `section`, l'étiquette même
        # que `_extend_last` laissait fausse. Une section B fusionnée dans un
        # chunk étiqueté A héritait donc de la DÉCISION DE PIED de A :
        #   · si A est dans `skip_footer_sections`, B perdait son garde-fou ;
        #   · si A n'y est pas, B en recevait un qui ne la concernait pas.
        # Le premier sens est le dangereux : un garde-fou qui disparaît ne se
        # voit pas. On décide donc sur TOUTES les sections contenues, et on
        # applique le pied dès qu'UNE seule le justifie — le sens sûr.
        sections = c["metadata"].get("sections") or [c["metadata"]["section"]]
        if any(_footer_applicable(s, skip_footer_sections) for s in sections):
            c["text"] += footer
            c["metadata"]["word_count"] = len(c["text"].split())
    return chunks


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
            # Toutes les sections réellement contenues, la première d'abord.
            # `section` reste l'étiquette PRIMAIRE et ne bouge pas : elle entre
            # dans le calcul de `id` (uuid5 de source::section::position), et la
            # changer re-identifierait silencieusement les chunks concernés au
            # prochain lot. Pour 0,5 % de chunks, ce n'est pas le bon échange.
            # L'honnêteté est portée par `sections` et par le titre inséré dans
            # le texte ; la stabilité par `section`.
            "sections": [section or "AUTRE"],
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
    # Réglage inerte : quand le pied est posé à la fin, on ne le passe pas à
    # `_mk_chunk` -- sinon il serait apposé deux fois.
    _pied_creation = None if PIED_POSE_A_LA_FIN else footer

    def _extend_last(extra: List[str], titre: Optional[str] = None) -> None:
        """Rattache `extra` au chunk précédent.

        ⚠️ DEUXIÈME DÉFAUT CONNU DE CETTE FONCTION, corrigé le 09/09.
        Elle mettait à jour le texte et le compte de mots, **jamais l'étiquette
        de section**. Un paragraphe écrit sous « ## Sujet B » se retrouvait dans
        un chunk étiqueté « Sujet A ». Mesuré par `feat/fiches` : 4 chunks
        omnibus sur 844 (0,5 %), 2 fiches sur 35, dont une où des chiffres
        AVA/VAE héritaient d'une étiquette « périmés à ne jamais citer ».

        C'est la même famille que le défaut du corpus juridique : **le contenu
        est juste, l'étiquette est fausse, et rien ne le signale.**

        Le premier défaut de cette même fonction — du contenu ajouté APRÈS le
        pied — a été corrigé le 07/09 par `PIED_POSE_A_LA_FIN`, et son
        commentaire est quatre-vingts lignes plus haut. Personne n'avait cherché
        le suivant. **Une fonction qui a déjà produit un défaut mérite qu'on
        cherche le suivant.**

        ⚠️ RIEN N'EST RÉINGÉRÉ. Ce correctif dort dans le code jusqu'au prochain
        lot : le corpus en base porte toujours les étiquettes fausses.

        ⚠️⚠️ ET CE N'EST PAS DU CODE DORMANT — correction de provenance du 09/09.
        J'avais écrit que « le carnet Colab produit le vrai corpus » : **c'est
        faux pour les fiches**. La chaîne réelle est
        `.github/workflows/rag-ingest.yml` → `scripts/ingest_lot.py` →
        `qe_rag/pipeline.py` → **ce fichier**. Les deux réingestions de la
        semaine (35 fiches, 881 points) sont passées par ici. Le défaut est donc
        **dans le corpus que l'utilisateur interroge aujourd'hui**, et le pied
        perdu (voir `_poser_pieds`) l'est pour de vrai.

        Le carnet a servi ailleurs — les codes, `QuestionParlementaire`. La
        divergence entre les deux implémentations (ici on fusionne avec une
        étiquette fausse, là-bas on JETTE la section courte) concerne donc des
        parties différentes du corpus, et reste à arbitrer.
        """
        c = chunks[-1]
        meta = c["metadata"]
        if titre and titre != "AUTRE" and titre not in meta["sections"]:
            # Le titre entre DANS LE TEXTE : c'est là que le modèle le lira, et
            # c'est ce qui rend la transition visible à la citation.
            c["text"] += f" {titre}. " + " ".join(extra)
            meta["sections"].append(titre)
        else:
            c["text"] += " " + " ".join(extra)
        meta["word_count"] = len(c["text"].split())

    for title, body in _sections(lines):
        words = carry + " ".join(_normalise_body(x) for x in body).split()
        carry = []
        if len(words) < min_words:
            if chunks:
                # Section ENTIÈRE rattachée au chunk précédent : c'est ici que
                # l'étiquette changeait de sens sans que rien ne le dise.
                _extend_last(words, title)
            else:
                carry = words                # rien avant : on reporte
            continue
        for debut, fin in _bornes_decoupe(words, max_words):
            piece = words[debut:fin]
            if len(piece) < min_words and chunks:
                # Morceau trop court DE LA MÊME SECTION : pas de changement
                # d'étiquette à signaler, sauf si le chunk précédent vient
                # d'une autre section -- `_extend_last` s'en charge.
                _extend_last(piece, title)
                continue
            head = f"{title}. " if title and title != "AUTRE" else ""
            chunks.append(_mk_chunk((head + " ".join(piece)).strip(), source, title, len(chunks),
                                    _pied_creation, skip_footer_sections))
            if len(chunks) >= max_chunks:
                return (_poser_pieds(chunks, footer, skip_footer_sections)
                        if PIED_POSE_A_LA_FIN else chunks)

    if carry:  # sections courtes en tête de document, jamais rattachées
        if chunks:
            _extend_last(carry)
        else:
            chunks.append(_mk_chunk(" ".join(carry), source, "AUTRE", 0,
                                    _pied_creation, skip_footer_sections))
    return (_poser_pieds(chunks, footer, skip_footer_sections)
            if PIED_POSE_A_LA_FIN else chunks)


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
