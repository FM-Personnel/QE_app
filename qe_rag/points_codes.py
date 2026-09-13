# -*- coding: utf-8 -*-
"""Construire les points d'un code juridique, SANS importer `app.py`.

Extrait de `scripts/essai_blanc_refonte_codes.py`, à l'identique. L'essai à
blanc importe désormais ce module au lieu d'en garder une copie : deux
implémentations d'une même règle divergent toujours, et on l'a payé deux fois
dans la nuit du 8 au 9 septembre (`qe_rag/chunk.py` contre le carnet, puis
`texte_de` contre `extract.py`).

Pourquoi séparer : l'essai à blanc importe `app.py` pour valider le contrat de
payload — ce qui fait se connecter le processus à Qdrant. Un carnet Colab qui
n'a que le modèle d'embedding à faire n'a aucune raison d'ouvrir une connexion
à la base, et n'a pas les secrets pour le faire. Le contrat, lui, a été validé
localement sur les 39 261 points le 09/09 ; il n'a pas à l'être une seconde fois
au moment d'encoder.

RIEN ICI N'ÉCRIT NULLE PART.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, List, Tuple

from qe_rag.chunk import _bornes_decoupe
from qe_rag.extract import _BALISE
from qe_rag.refs_legislatives import extract_base_legislative

# Le modèle dense tronque à 512 tokens. Un article plus long n'est pas encodé
# en entier : c'est la SEULE raison de le découper. ~1,35 token par mot en
# français -> 380 mots. Réglage, pas vérité.
MAX_MOTS = 380

_ESPACES = re.compile(r"[ \t ]+")

CODE_VERS_COLLECTION = {
    "CASF": "CASF",
    "CT": "Code du travail",
    "CSP": "Code de la santé publique",
    "CSS": "Code de la sécurité sociale",
}


def texte_de(html_brut: str) -> str:
    """HTML -> texte. Les fins de bloc deviennent des sauts de ligne, pour que
    le découpage aux phrases voie les frontières d'alinéa."""
    s = re.sub(r"(?i)</(p|div|li|tr|h\d)>", "\n", html_brut or "")
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = _BALISE.sub(" ", s)
    s = html.unescape(s)
    s = _ESPACES.sub(" ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return "\n".join(l.strip() for l in s.split("\n")).strip()


def formes_du_numero(num: str, computed=None) -> List[str]:
    """Les formes sous lesquelles le numéro s'écrit — variante C, mesurée
    significative (6 gains, 0 perte, p = 0,031)."""
    formes = {num}
    for c in (computed or []):
        if c:
            formes.add(str(c))
    m = re.match(r"([LRD])(\d.*)$", num or "")
    if m:
        formes.add(f"{m.group(1)}. {m.group(2)}")
        formes.add(m.group(2))
    return sorted(formes)


def parcourir(noeud, chemin: Tuple[str, ...] = ()) -> Iterable[tuple]:
    """Rend (article, chemin de sections). Deux formes d'entrée acceptées.

    - l'ARBRE `legiPart` : {"articles": [...], "sections": [...]} ;
    - la MOISSON PAR ARBRE : une LISTE PLATE d'articles portant déjà leur
      `contexte_hierarchique`, qu'on redécoupe plutôt que de reconstruire un
      arbre — moins de code, aucune occasion de diverger.
    """
    if isinstance(noeud, list):
        for a in noeud:
            ctx = a.get("contexte_hierarchique") or ""
            yield a, tuple(x.strip() for x in ctx.split(">") if x.strip())
        return
    for a in (noeud.get("articles") or []):
        yield a, chemin
    for s in (noeud.get("sections") or []):
        titre = s.get("title") or s.get("titre") or ""
        yield from parcourir(s, chemin + (titre,))


def niveau(chemin: Tuple[str, ...], mot: str) -> str:
    """Le maillon du chemin qui commence par « Livre », « Titre », etc."""
    for c in chemin:
        if c.lower().startswith(mot):
            return c
    return ""


def construire(source, code: str, date_version: str, max_mots: int = MAX_MOTS):
    """Rend (points, statistiques). Aucune écriture, aucune connexion."""
    collection = CODE_VERS_COLLECTION[code]
    points, splits, pertes_html, sans_num = [], 0, [], 0

    for art, chemin in parcourir(source):
        num = (art.get("num") or "").strip()
        if not num:
            sans_num += 1
            continue
        brut = art.get("content") or art.get("texte") or ""
        texte = texte_de(brut)
        if not texte:
            continue
        pertes_html.append(len(_BALISE.sub("", brut)) - len(texte))

        contexte = " > ".join(x for x in chemin if x)
        nums = formes_du_numero(num, art.get("computedNums"))
        refs = extract_base_legislative(texte, collection)
        mots = texte.split()
        bornes = _bornes_decoupe(mots, max_mots)
        if len(bornes) > 1:
            splits += 1
        for rang, (d, f_) in enumerate(bornes):
            points.append({
                "chunk_id": f"{num}_chunk{rang}",
                "num": num,
                "titre": f"Article {num}",
                "contenu": " ".join(mots[d:f_]),
                "article_complet": texte,
                "contexte_hierarchique": contexte,
                "collection": collection,
                "partie": niveau(chemin, "partie"),
                "livre": niveau(chemin, "livre"),
                "titre_structure": niveau(chemin, "titre"),
                "chapitre": niveau(chemin, "chapitre"),
                "section": niveau(chemin, "section"),
                "sous_section": niveau(chemin, "sous-section"),
                "paragraphe": niveau(chemin, "paragraphe"),
                # La moisson enrichie du 09/09 le collecte ; l'arbre `legiPart`
                # ne le portait pas, d'où la valeur vide en repli.
                "sous_paragraphe": art.get("sous_paragraphe") or niveau(chemin, "sous-paragraphe"),
                "base_legislative": refs,
                "code": code,
                "article_id": art.get("article_id") or art.get("cid") or art.get("id") or "",
                "nums_normalises": nums,
                "date_version_debut": art.get("dateDebut") or "",
                "date_version_fin": art.get("dateFin") or "",
                "etat": art.get("etat") or "",
                "rang_dans_article": f"{rang+1}/{len(bornes)}",
                # L'ABSENCE de ce champ marque l'ancien corpus : rien à
                # rétro-taguer pour distinguer les deux régimes.
                "origine_corpus": f"api_legifrance/{date_version}",
                "nb_liens_citation": (art.get("nb_liens_citation")
                                      if art.get("nb_liens_citation") is not None
                                      else len(art.get("lstLienCitation") or [])),
                "nb_liens_modification": (art.get("nb_liens_modification")
                                          if art.get("nb_liens_modification") is not None
                                          else len(art.get("lstLienModification") or [])),
            })

    stats = {
        "articles": len({p["num"] for p in points}),
        "points": len(points),
        "articles_decoupes": splits,
        "sans_numero": sans_num,
        "perte_html_mediane": (sorted(pertes_html)[len(pertes_html) // 2]
                               if pertes_html else 0),
    }
    return points, stats
