"""Extraction des renvois legislatifs — PORT VERBATIM du carnet d'origine.

⚠️ Cette fonction est recopiee TELLE QUELLE depuis
`reference/codes_sources/carnets/Clean_&_chunk_&_embed_Codes_index_v4.ipynb`
(cellule d'upload Qdrant). Elle produit le champ `base_legislative` du payload.

NE PAS « AMELIORER » SANS MESURER. La contrainte 1 de la refonte est que le
payload reste compatible avec le retrieval actuel : si cette fonction change,
`base_legislative` ne veut plus dire la meme chose qu'aujourd'hui et la
comparaison avant/apres perd son bras de controle.

Ce qu'elle capture : les renvois ECRITS DANS LE TEXTE de l'article.
Ce qu'elle ne capture pas : le graphe editorial de Legifrance (codification,
concordance avec les codes abroges) -- verifie le 08/09, les deux ensembles ne
se recouvrent pas et aucun n'est un sur-ensemble de l'autre. Voir
`reference/conception_refonte_codes.md` §3.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


# Dependance du carnet, portee VERBATIM elle aussi. Elle leve sur un
# numero a plus d'un tiret (« 312-194-13 ») ou non numerique -- c'est le
# comportement d'origine, et l'appelant l'attrape. On ne le « corrige »
# pas : ce serait changer ce que `base_legislative` contient aujourd'hui.
def parse_num(num_str: str) -> tuple:
    """Sépare le numéro principal et le suffixe (ex: '123-4' -> (123, 4))."""
    if '-' in num_str:
        main, suffix = num_str.split('-')
        return int(main), int(suffix)
    return int(num_str), None


def extract_base_legislative(text: str, current_collection: str) -> Optional[List[Dict[str, str]]]:
    """
    Extrait toutes les références législatives d'un texte, avec gestion des collections.
    Args:
        text: Texte de l'article.
        current_collection: Collection par défaut (ex: "CASF").
    Returns:
        Liste de dictionnaires {"uid": "L123-4", "collection": "CSP"}, ou None si aucune référence.
    """
    # Nettoyage du texte : suppression des retours à la ligne et normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()

    references = []
    seen = set()

    # Mapping des noms de codes vers les collections (expressions complètes)
    code_mapping = {
        "code du travail": "Code du travail",
        "code de la santé publique": "Code de la santé publique",
        "code de la sécurité sociale": "Code de la sécurité sociale",
        "code de l'action sociale et des familles": "CASF",
    }

    # Dernière collection explicitement mentionnée
    last_explicit_collection = current_collection

    # Patterns pour capturer les références (ordre important : explicites en premier)
    patterns = [
        # 1. Références avec code explicite (priorité maximale) - CORRIGÉE POUR "aux articles"
        (r'(?:aux|de l\'?article|des|les\s*)\s*articles?\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)\s*(?:à\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?))?\s*du\s*(?:code\s*)?(?:de\s+)?(?:la\s+)?(code du travail|code de la santé publique|code de la sécurité sociale|code de l\'?action sociale et des familles)', "explicit"),
        # 2. Plages : "articles R. 821-1 à R. 821-10"
        (r'(?:articles?|art\.?)\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)\s*à\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)', None),
        # 3. Listes : "articles R. 821-1, R. 821-2, R. 821-3"
        (r'(?:articles?|art\.?)\s*((?:[LRD]\.\s*\d{1,4}(?:-\d{1,4})?\s*(?:,|et|ou)\s*)+[LRD]\.\s*\d{1,4}(?:-\d{1,4})?)', None),
        # 4. Références simples : "article R. 821-1" ou "R. 821-1"
        (r'(?:article|art\.?)\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)|([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)', None),
        # 5. "du même code" (réutilise la dernière collection mentionnée)
        (r'de\s*(?:l\'?article|les\s*articles?)\s*([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)\s*du\s*même\s*code', "same"),
    ]

    # --- Première passe : traiter les références explicites pour mettre à jour last_explicit_collection ---
    for pattern, pattern_type in patterns:
        if pattern_type == "explicit":
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    type_start, num_start, type_end, num_end, code_name = match.groups()
                    if not type_end:
                        type_end = type_start
                    uid = f"{type_start}{num_start}"
                    collection = code_mapping.get(code_name.lower(), current_collection)
                    last_explicit_collection = collection  # Met à jour la collection pour les plages/listes
                    if uid not in seen:
                        seen.add(uid)
                        references.append({"uid": uid, "collection": collection})
                except (ValueError, AttributeError):
                    continue

    # --- Deuxième passe : traiter plages, listes et références simples avec la collection mise à jour ---
    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                if pattern_type is None:  # Plages ou listes
                    if pattern == patterns[1][0]:  # Plage : "R. 821-1 à R. 821-10"
                        type_start, num_start, type_end, num_end = match.groups()
                        if not type_end:
                            type_end = type_start
                        start_num, start_suffix = parse_num(num_start)
                        end_num, end_suffix = parse_num(num_end)

                        # Génération des UIDs pour la plage
                        if start_suffix and end_suffix:  # Plage de suffixes (ex: R. 821-1 à R. 821-10)
                            for s in range(int(start_suffix), int(end_suffix) + 1):
                                uid = f"{type_start}{start_num}-{s}"
                                if uid not in seen:
                                    seen.add(uid)
                                    references.append({"uid": uid, "collection": last_explicit_collection})
                        else:  # Plage de numéros (ex: R. 123-4 à R. 125-6)
                            for i in range(int(start_num), int(end_num) + 1):
                                uid = f"{type_start}{i}"
                                if uid not in seen:
                                    seen.add(uid)
                                    references.append({"uid": uid, "collection": last_explicit_collection})

                    elif pattern == patterns[2][0]:  # Liste : "R. 821-1, R. 821-2, R. 821-3"
                        if match.group(1):  # Vérification NoneType
                            refs = re.split(r'\s*(?:,|et|ou)\s*', match.group(1))
                            for ref in refs:
                                ref_match = re.match(r'([LRD])\.\s*(\d{1,4}(?:-\d{1,4})?)', ref)
                                if ref_match:
                                    type_, num = ref_match.groups()
                                    uid = f"{type_}{num}"
                                    if uid not in seen:
                                        seen.add(uid)
                                        references.append({"uid": uid, "collection": last_explicit_collection})

                    elif pattern == patterns[3][0]:  # Référence simple
                        type_ = match.group(1) or match.group(3)
                        num = match.group(2) or match.group(4)
                        uid = f"{type_}{num}"
                        if uid not in seen:
                            seen.add(uid)
                            references.append({"uid": uid, "collection": last_explicit_collection})

                elif pattern_type == "same":  # "du même code"
                    type_, num = match.groups()
                    uid = f"{type_}{num}"
                    if uid not in seen:
                        seen.add(uid)
                        references.append({"uid": uid, "collection": last_explicit_collection})

            except (ValueError, AttributeError):
                continue

    return references if references else None
