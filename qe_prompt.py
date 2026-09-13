# -*- coding: utf-8 -*-
"""LE PROMPT DE LA REPONSE PARLEMENTAIRE -- gabarit, reglages et profil.

Extrait d'`app.py` le 13/09/2026, SANS CHANGER UN OCTET DU PROMPT.

POURQUOI CE FICHIER EXISTE
==========================
Le harnais d'evaluation (`feat/eval`) reconstruisait le prompt de son cote. Cinq
divergences harnais/production ont ete trouvees en une semaine, dont deux qui
faussaient des mesures deja publiees : une cible de longueur de 500-800 mots la
ou la production demande 250-350, et un modele `small` la ou la production
emploie `large`.

La regle du projet est qu'on IMPORTE au lieu de RECOPIER, parce que deux
implementations d'une meme regle divergent toujours. Mais le harnais ne pouvait
pas importer `app.py` : son integration continue n'installe que `requests` et
`fastembed`, et exige ZERO EFFET DE BORD AU CHARGEMENT -- or importer `app.py`
leve sans trois secrets, charge deux modeles et APPELLE `get_collections()`, un
appel reseau, ligne 264.

D'ou ce module. Il n'importe que la bibliotheque standard et `pytz`. Aucun
secret, aucun reseau, aucun modele, aucun `streamlit`. Un `pip install pytz`
suffit a s'en servir.

UN SEUL FICHIER, PAS UN PAQUET -- et c'est une decision de deploiement, pas de
style : un fichier plat ne peut pas etre PARTIELLEMENT deploye. `qe_rag/` l'a
ete (il manquait `refs_legislatives.py`, et personne ne l'a vu parce que rien
dans l'arbre deploye ne l'importait). Un paquet offre cette prise, un fichier non.

⚠️ LES REGLAGES VIVENT ICI, ET NULLE PART AILLEURS
`POSITIONS_ORALES`, `ABSTENTION_POSITION`, `VIVIER_OUVERTURES`,
`GRAINE_OUVERTURES` sont lus par les fonctions de ce module, donc dans LES
GLOBALES DE CE MODULE. Les basculer ailleurs (`app.POSITIONS_ORALES = True`) ne
produirait AUCUN effet -- et c'est le pire cas, parce qu'un test ecrit ainsi ne
tombe pas : il passe A VIDE. Pour les basculer, c'est ici :

    import qe_prompt
    qe_prompt.POSITIONS_ORALES = True

`app.py` ne les redeclare pas et les lit par `qe_prompt.X`. Un temoin du smoke
verifie que `app` ne porte AUCUN de ces quatre noms, pour que l'ancienne voie
echoue bruyamment au lieu de ne rien faire.

LE TEMOIN DE L'EXTRACTION : l'empreinte du prompt sur 36 cas fixes, date
neutralisee (`scripts/empreinte_prompt.py`), IDENTIQUE avant et apres --
e1e29709cb972b93561ae84cfaf288e0148fcf6545ab9c257706a5914e04c007.
"""
from __future__ import annotations

import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytz

# =====================================================================
# POSITIONS EXPRIMÉES EN SÉANCE — INERTE (`POSITIONS_ORALES`)
# ---------------------------------------------------------------------
# Une réponse orale N'EST PAS un modèle de rédaction : c'est pour cela que 883
# transcriptions de séance ont été supprimées de `QuestionParlementaire` le
# 10/09. Elle n'entre donc PAS dans le CONTEXTE PARLEMENTAIRE, dont la consigne
# dit qu'il sert « au registre, au ton et aux axes » et que ses chiffres sont
# « datés, donc présumés périmés ».
#
# Elle entre dans un bloc PROPRE, après RECHERCHE INTERNET — même famille,
# « daté et sourcé » —, comme une SOURCE DATÉE À CITER : date de séance,
# chambre, orateur nommé, position, lien.
#
# BLOCAGE LEVÉ LE 11/09 — les vecteurs portent bien le champ `question`, donc
# comparer l'embedding de la question de l'utilisateur a un sens. La source est
# le workflow d'ingestion lui-même, `.github/workflows/ingest-oraux.yml` :
# `modele.encode([p["question"] for p in points])`. Ni `reponse`, ni la
# concaténation — même convention que `QuestionParlementaire`.
#
# ⚠️ À RETENIR : ma mesure de corrélation classait `question+reponse` EN TÊTE
# (+0,346 contre +0,248), c'est-à-dire **pas** le champ réellement encodé. Son
# seuil (écart < 0,05) a refusé de conclure, et c'est la seule raison pour
# laquelle on n'a pas retenu la mauvaise réponse : un maximum de corrélation
# n'identifie pas une source. Détail : `reference/conception_positions_orales.md`.
#
# Ce qui reste avant activation n'est plus une inconnue mais une décision : le
# réglage change le prompt en production, donc il s'active sur accord explicite.
# =====================================================================
POSITIONS_ORALES = False

# --- Plafonds de génération (max_tokens) ---------------------------------
# Validés par l'utilisateur le 06/09 sur le chiffrage de `app_findings.md` § F10.
#
# PRINCIPE : le plafond est un GARDE-FOU CONTRE L'EMBALLEMENT, pas un second
# régulateur de longueur. La longueur est pilotée par la consigne du prompt
# (`_LONGUEUR_MOTS`). Tant que le plafond était au ras de la cible, c'est LUI
# qui terminait la réponse au lieu de la composition du modèle — d'où des
# coupures en plein mot : les 3 occurrences de F10 sont toutes en « Moyenne »,
# précisément la longueur où la marge était nulle.
#
# CALIBRAGE : ~1,6 à 1,8 fois le besoin au HAUT de la fourchette, avec un besoin
# estimé à 1,63-1,71 token par mot (6,82 caractères/mot mesurés sur 55 réponses
# réelles ; 4,0 à 4,19 caractères/token).
#
#   longueur   cible (mots)   besoin au plafond de cible   avant   après
#   Courte     250-350        570-597                      500     1000
#   Moyenne    450-600        977-1023                     1000    1800
#   Longue     900-1200       1953-2046                    2200    3200
_MAX_TOKENS_PAR_LONGUEUR = {"Courte": 1000, "Moyenne": 1800, "Longue": 3200}

# Fenêtre de contexte du modèle (prompt + réponse). Sert à réserver la place de
# la réponse dans le garde-fou de taille de prompt.
FENETRE_MODELE = {"large": 32000, "autre": 16000}

# SORTI DE `generate_response` LE 13/09, SANS CHANGER UNE VALEUR. Motif : il
# etait declare A L'INTERIEUR de la fonction, donc INIMPORTABLE -- et le
# harnais d'evaluation en a besoin pour reproduire les memes blocs. Une valeur
# que deux outils doivent partager ne peut pas vivre dans une portee locale,
# sinon le second en garde une copie et les deux divergent.
#
# ⚠️ Il manque un budget a DEUX blocs : `uploaded_documents` et
# `positions_orales` n'ont AUCUNE entree ici (voir le commentaire d'app.py
# vers la ligne 4337 : « plus gros bloc -- n'a aucun budget dans
# TOKEN_LIMITS »). Qui lit cette table pour reproduire l'assemblage doit le
# savoir : quatre blocs sont plafonnes ici, deux le sont ailleurs.
# Limites de tokens pour Mistral Large
#
# ⚠️ CE NE SONT PAS DES PLAFONDS DE BLOC. Chaque valeur borne le seul
# CORPS tronqué — `reponse` côté parlementaire, `contenu` côté juridique.
# L'en-tête, le titre, le texte de la question et les annotations sont
# ajoutés par-dessus et n'entrent dans aucun budget. Mesuré : le bloc
# juridique dépasse son plafond sur 37 questions sur 40 en `medium`
# (+1 188 tokens en médiane), le parlementaire sur 30 sur 40.
#
# `parliamentary_context` : 4 500 -> 1 500, décidé par l'utilisateur le
# 06/09 sur un test à l'aveugle de `feat/eval` — 7 paires anonymisées,
# contexte complet contre contexte réduit : 3 préférences pour l'un,
# 3 pour l'autre, 1 égalité, toutes qualifiées de « légères ».
# `medium`/`small` passent AUSSI à 1 500, et non à une valeur réduite
# proportionnellement : le plafond est divisé par 3 réponses, si bien
# que 1 500 rend exactement les 500 tokens par réponse qui ont été
# ÉPROUVÉS. Une réduction proportionnelle (1 000) descendrait à 333 par
# réponse, en deçà de ce que le test a validé.
#
# ⚠️ Décision de l'utilisateur : ce test sera REFAIT dans quelques
# semaines, quand l'outil aura évolué. La valeur vaut pour l'état
# actuel, pas pour toujours.
TOKEN_LIMITS = {
    "large": {
        "question": 1500,
        "parliamentary_context": 1500,  # 3 réponses QE, 500 tokens each
        "search_context": 3000,         # 15 snippets Google
        "legal_context": 5000,          # À affiner plus tard
    },
    "medium": {
        "question": 1000,
        "parliamentary_context": 1500,  # idem : 500 tokens par réponse
        "search_context": 1500,         # 10 snippets Google
        "legal_context": 2000,          # À affiner plus tard
    },
    "small": {
        "question": 1000,               # Même valeur que medium
        "parliamentary_context": 1500,  # Même que medium
        "search_context": 1500,         # 10 snippets Google (même que medium)
        "legal_context": 2000,          # Même que medium
    }
}

def _cle_longueur(longueur: str) -> str:
    """« Moyenne (500 mots) » -> « Moyenne ». Repli sur Moyenne si inconnu."""
    for cle in _MAX_TOKENS_PAR_LONGUEUR:
        if (longueur or "").startswith(cle):
            return cle
    return "Moyenne"

# --- F16 : dégrader le contexte plutôt que refuser la génération -------------
# Quand le prompt dépasse `fenêtre − max_tokens`, l'app rendait une ERREUR et
# aucune réponse (#8167). Une réponse sur contexte réduit vaut mieux que pas de
# réponse : on retire du contexte jusqu'à tenir, dans un ordre de sacrifice.
#
# CET ORDRE N'EST PAS UNE INTUITION : il est déduit de ce que le prompt dit
# lui-même de chaque bloc.
#   1. CONTEXTE PARLEMENTAIRE — le système le déclare utile « d'abord au
#      registre, au ton et aux axes » et présume ses chiffres périmés. C'est le
#      bloc dont on perd le moins en le coupant.
#   2. RECHERCHE INTERNET — daté et sourcé, mais complémentaire.
#   3. DOCUMENTS DE RÉFÉRENCE — déjà trié par score : couper la queue retire les
#      extraits les plus faibles. Vient après les deux précédents parce que le
#      système le désigne comme « la source à privilégier pour tout chiffre ».
#   4. TEXTES JURIDIQUES — le prompt les dit « prioritaires en cas de
#      contradiction ». Sacrifiés en dernier.
# La QUESTION n'est jamais réduite ici : elle a déjà son plafond propre.
#
# DEUX GARANTIES, demandées par Coordination et vérifiées par le smoke :
#   · le chemin normal n'est PAS touché — si le prompt tient, on sort avant
#     d'avoir rien modifié, et la trace reste vide ;
#   · `uploaded_documents` ne reçoit AUCUN budget permanent. Lui en donner un
#     exigerait de choisir un nombre sans mesure — ce qui vient précisément de
#     nous coûter une régression. Ici il n'est borné QUE pendant la dégradation,
#     donc il ne peut pas mordre sur un cas qui fonctionne : c'est vrai par
#     construction, pas par calibrage.
# =====================================================================
# LA PRIORITÉ DES BLOCS, DÉCLARÉE UNE SEULE FOIS  (chantier D)
# ---------------------------------------------------------------------
# Avant : la priorité entre sources vivait à DEUX endroits qui pouvaient
# diverger — l'ordre des blocs dans le f-string du prompt, et `_ORDRE_SACRIFICE`
# qui décide lequel est amputé quand la fenêtre sature. Deux encodages de la
# même notion, et une règle écrite à deux endroits finit par diverger en
# silence : ça s'est produit deux fois le 11/09 (l'écran et le prompt pour les
# collections ; la clé d'identité et son test de présence).
#
# ⚠️ CE QUE CETTE DÉCLARATION N'UNIFIE PAS, ET C'EST VOULU.
# Il y a DEUX ordres distincts, pas un :
#   · `rang_prompt`     — où le bloc APPARAÎT dans le prompt (ordre de lecture) ;
#   · `rang_protection` — ce qui SURVIT quand la fenêtre sature.
# Ils n'ont aucune raison de coïncider, et les fondre en un seul nombre
# produirait une abstraction qui n'est vraie d'aucun des deux. Une troisième
# priorité existe encore — l'ordre des ARTICLES à l'intérieur du bloc juridique
# (`sort_articles_for_prompt`, par provenance) — et elle reste DEHORS : elle
# ordonne des articles, pas des blocs. C'est un autre axe.
#
# `criteres` = les critères de la liste ordonnée de l'utilisateur (10/09) que le
# bloc sert, du plus prioritaire au moins. C'est un TUPLE et non un seul nombre :
# un bloc en sert souvent plusieurs, et n'en retenir qu'un a failli me faire
# écrire que RECHERCHE INTERNET ne sert pas le critère 1 — alors que son en-tête
# dit littéralement « actualités et positions du Gouvernement ». Le rang qui
# compte pour une comparaison est le PLUS HAUT, donc `min()`.
# Ils sont déclarés pour que la divergence entre ce que nous protégeons et ce
# que l'utilisateur a classé soit LISIBLE — voir `divergences_priorite()`.
# Cette déclaration ne CHANGE aucune priorité : elle les rend visibles. Les
# changer est un arbitrage, pas un remaniement.
#
# Chaque bloc : (clé dans le dict de contextes, étiquette de trace,
#                en-tête dans le prompt, rang_prompt, rang_protection, critère)
# `rang_protection` : 1 = sacrifié EN PREMIER, 5 = protégé jusqu'au bout.
_BLOCS_PROMPT = (
    ("parliamentary_context", "contexte parlementaire",
     "CONTEXTE PARLEMENTAIRE", 1, 2, (5,)),
    ("legal_context", "textes juridiques",
     "TEXTES JURIDIQUES APPLICABLES", 2, 5, (4,)),
    ("uploaded_documents", "documents de référence",
     "DOCUMENTS DE RÉFÉRENCE", 3, 4, (4,)),
    ("search_context", "recherche internet",
     "RECHERCHE INTERNET", 4, 3, (1, 4)),
    # POSITIONS EXPRIMÉES EN SÉANCE — inerte (`POSITIONS_ORALES`). Absent du
    # dictionnaire tant que le réglage l'est : `blocs.get` rend "" et la boucle
    # passe, donc aucune trace et aucun effet.
    #
    # ⚠️ Son `rang_protection` de 1 est celui que je lui ai donné le 11/09, au
    # motif qu'il était « l'ajout le plus récent et le moins portant ». La liste
    # ordonnée de l'utilisateur dit l'inverse : ce bloc est la seule source du
    # critère 1, LE SEUL QUI DISQUALIFIE. Le rang est donc probablement faux —
    # mais le corriger est un ARBITRAGE, pas un remaniement, et il se décide
    # avec l'activation du réglage puisque c'est la même question.
    ("positions_orales", "positions en séance",
     "POSITIONS EXPRIMÉES EN SÉANCE", 5, 1, (1,)),
    # =================================================================
    # LA SAISIE DU RÉDACTEUR — DÉCLARÉE, PAS ENCORE RENDUE
    # -----------------------------------------------------------------
    # `entete = None` signifie : ce bloc a une PLACE, il n'a pas encore de
    # FORME. C'est la distinction exacte du mandat — la place est un rang, la
    # forme est un objet — et `ordre_des_entetes()` saute les blocs non rendus.
    #
    # POURQUOI IL EXISTE : le critère 3 demande d'annoncer qu'un sujet va être
    # traité. Aucune source publique ne porte cette information, et
    # l'utilisateur a dit qu'il en disposerait nécessairement au moment de
    # rédiger. C'est donc une ENTRÉE, pas une récupération — et la seule source
    # du système dont la fiabilité soit établie par l'utilisateur lui-même.
    #
    # `rang_protection = 6`, le plus haut, et ce n'est pas un arbitrage laissé
    # ouvert : sacrifier ce que l'utilisateur a écrit lui-même pour garder ce
    # qu'un moteur a retrouvé serait absurde dans tous les cas de figure.
    #
    # `rang_prompt = 6` est PROVISOIRE au sens strict : il ne peut pas être
    # faux tant que le bloc n'est pas rendu, et il sera décidé avec la forme.
    #
    # ⚠️ CE QUI NE PEUT PAS SE DÉCIDER SANS LA FORME, et je m'arrête là plutôt
    # que de choisir : `criteres` vaut `(3,)`. Si le champ admet aussi que le
    # rédacteur ÉNONCE une position (« le Gouvernement s'oppose à … »), alors il
    # sert aussi le critère 1 — et ce n'est pas cosmétique : `blocs_du_critere(1)`
    # commande le déclenchement de l'abstention (C1). Un champ rempli qui ne
    # compterait pas comme source de position ferait s'abstenir le modèle alors
    # que le rédacteur vient de lui donner la réponse. La forme décide donc du
    # critère, et le critère décide d'un comportement : les deux ne se séparent
    # pas. Voir `reference/saisie_redacteur_place.md`.
    ("saisie_redacteur", "information du rédacteur",
     None, 6, 6, (3,)),
)

# DÉRIVÉ, jamais récrit. Le moins protégé d'abord.
_ORDRE_SACRIFICE = tuple(
    (cle, etiquette) for cle, etiquette, _e, _rp, _prot, _c
    in sorted(_BLOCS_PROMPT, key=lambda b: b[4])
)

def ordre_des_entetes() -> list:
    """Les en-têtes de blocs dans l'ordre où le prompt doit les présenter.

    Sert au témoin de parité : le f-string du prompt reste littéral — le
    réécrire pour le dériver changerait le prompt à l'octet, ce qu'on s'interdit
    ici — mais un test compare l'ordre réellement produit à CETTE liste. La
    divergence devient donc impossible sans qu'un test tombe, ce qui est la
    garantie qu'on cherchait ; la dérivation textuelle ne l'est pas.
    """
    return [e for _c, _et, e, _rp, _prot, _cr
            in sorted(_BLOCS_PROMPT, key=lambda b: b[3]) if e is not None]

def divergences_priorite() -> list:
    """Où ce que nous PROTÉGEONS contredit ce que l'utilisateur a CLASSÉ.

    Rend une liste de couples (mieux_protégé, moins_protégé) dont le critère
    servi est pourtant moins prioritaire — critère 1 étant le plus haut. Une
    liste non vide n'est pas un défaut de code : c'est une question d'arbitrage,
    et l'intérêt de la déclaration est qu'elle la rende lisible au lieu de la
    laisser dans deux mécanismes.

    ⚠️ CE QUE CETTE FONCTION NE SAIT PAS DIRE, découvert en déclarant la saisie
    du rédacteur le 12/09. Elle compare DEUX axes — protection et critère servi
    — alors qu'il en existe un TROISIÈME : la FIABILITÉ de la source. La saisie
    du rédacteur sert le critère 3, donc elle apparaît ici comme « trop
    protégée » face aux sources du critère 1 ; or elle est la seule source dont
    la fiabilité soit établie par l'utilisateur lui-même, et la sacrifier pour
    garder ce qu'un moteur a retrouvé n'aurait de sens dans aucun cas.

    Je ne complète PAS la fonction d'un troisième axe : je n'ai aucune mesure
    pour ordonner les fiabilités, et un axe inventé serait pire que l'axe
    manquant. Les couples qui viennent de `saisie_redacteur` sont donc des
    divergences PAR CONSTRUCTION, à lire comme telles et non comme des
    arbitrages — le témoin de smoke les sépare des deux vraies.
    """
    out = []
    for cle_a, _ea, _ha, _pa, prot_a, crits_a in _BLOCS_PROMPT:
        for cle_b, _eb, _hb, _pb, prot_b, crits_b in _BLOCS_PROMPT:
            if prot_a > prot_b and min(crits_a) > min(crits_b):
                out.append((cle_a, cle_b))
    return out

# =====================================================================
# C1 — L'ABSTENTION : dire qu'on ne sait pas, au lieu d'inférer  (INERTE)
# ---------------------------------------------------------------------
# « Un système qui ne peut pas disqualifier sa propre réponse ne peut pas
# respecter un critère disqualifiant. » L'application sait dire « la recherche
# juridique a ÉCHOUÉ » ; elle ne sait pas dire « je ne connais pas la position
# du Gouvernement sur ce point ». Or le critère 1 DISQUALIFIE : se taire y vaut
# mieux qu'affirmer.
#
# ⚠️ OÙ CETTE RÈGLE VIT, puisque Coordination demande qu'elle soit déclarée et
# non tenue de tête : PAS dans `_BLOCS_PROMPT`. L'abstention n'est pas un bloc
# de source, c'est une règle sur ce qui se passe quand les sources d'un critère
# sont toutes vides. La ranger parmi les blocs serait la sur-unification qu'on
# vient d'écarter deux fois. Elle a donc sa déclaration propre — mais le lien
# critère → blocs est DÉRIVÉ de `_BLOCS_PROMPT` par `blocs_du_critere()`, et
# n'est jamais réécrit ici. Sa PLACE dans le prompt est le bloc CONSIGNES DE
# RÉDACTION, qui est l'endroit des instructions.
ABSTENTION_POSITION = False

# ⚠️ UN BLOC VIDE NE L'EST PAS LITTÉRALEMENT. Quand rien n'est trouvé, les blocs
# reçoivent une PHRASE SENTINELLE (`app.py` ~4461, ~4481, ~4556). Un test de
# vérité naïf (`if search_context:`) serait donc TOUJOURS vrai, l'abstention ne
# se déclencherait jamais, et le contrôle aurait l'air de passer tout en ne
# faisant rien — exactement le défaut d'instrument qu'on paie depuis une
# semaine. Les sentinelles sont énumérées ici, et un témoin de smoke vérifie que
# chacune existe encore TELLE QUELLE dans ce fichier : en renommer une fait
# tomber le test au lieu de désactiver l'abstention en silence.
_SENTINELLES_VIDE = (
    "Aucun contexte parlementaire trouvé.",
    "Aucun texte juridique spécifique n'a été identifié.",
    "Aucune recherche internet effectuée.",
    # ⚠️ AJOUTÉE LE 12/09, ET ELLE MANQUAIT. `formater_recherche_internet` rend
    # cette phrase quand aucun résultat n'est exploitable. Elle fait 36
    # caractères : mon plancher de 40 la rattrapait PAR CHANCE, à quatre
    # caractères près. Une reformulation un peu plus longue aurait rouvert le
    # trou sans qu'aucun test ne tombe.
    "Aucun résultat internet exploitable.",
    # ⚠️ La phrase de PANNE n'est PAS ici : elle est dans
    # `_ANNOTATIONS_SANS_CONTENU`, parce qu'elle doit être retirée jusqu'à la
    # fin du paragraphe et non par correspondance exacte. L'avoir mise dans LES
    # DEUX listes cassait le mécanisme : la boucle des sentinelles retirait le
    # préfixe, après quoi la boucle des annotations ne trouvait plus son amorce
    # et laissait le reste de la phrase — qui repassait le plancher. Deux
    # mécanismes corrects qui, appliqués l'un après l'autre, s'annulent.
)

# ⚠️ ET CE QUI M'AVAIT ÉCHAPPÉ COMPLÈTEMENT : un bloc vide peut être vide ET
# ANNOTÉ. `search_context` reçoit en plus « (Peu de résultats pertinents : N
# résultat(s) écarté(s)…) », et `_reduire_bloc` ajoute « (Bloc réduit : …) ».
# Un bloc valant « Aucun résultat internet exploitable. » + la note fait 125
# caractères : mon test « contient une sentinelle → vide » échouait, et le bloc
# comptait comme une SOURCE DE POSITION. C'est-à-dire que l'abstention ne se
# déclenchait pas dans le cas exact où elle existe.
#
# D'où le changement de principe : on RETIRE tout ce qui n'est pas du contenu,
# puis on regarde ce qui reste. « Contenir une sentinelle » ne compose pas ;
# « retirer puis mesurer le reste » compose.
_ANNOTATIONS_SANS_CONTENU = (
    "Peu de résultats pertinents",
    "Bloc réduit",
    # ⚠️ ICI ET NON DANS `_SENTINELLES_VIDE`, et la nuance a coûté un test rouge.
    # Une sentinelle est retirée par correspondance EXACTE ; or cette phrase
    # porte un motif variable et une phrase d'explication. La déclarer comme
    # sentinelle n'en retirait que le PRÉFIXE, et le reste — « Ce n'est pas une
    # absence de résultat : on ignore… » — repassait le plancher de substance,
    # donc comptait comme une SOURCE DE POSITION. Retirée jusqu'à la fin du
    # paragraphe, comme les annotations.
    "La recherche internet n'a PAS PU avoir lieu",
)
# Plancher de substance : sous ce seuil, un bloc ne porte rien d'exploitable
# (les doublures de test passent « — »). Ceinture ET bretelles avec les
# sentinelles, parce qu'une sentinelle neuve ajoutée ailleurs ne serait pas ici.
_MIN_CAR_BLOC_UTILE = 40

_ABSTENTION_PAR_CRITERE = {
    1: ("Aucune source fournie n'établit la position du Gouvernement sur le "
        "point soulevé. N'en affirmez aucune et n'en déduisez aucune : "
        "rappelez le cadre applicable, et indiquez que le sujet fait l'objet "
        "d'un examen, sans prêter au Gouvernement une orientation que rien "
        "ici n'atteste."),
}

def bloc_porte_quelque_chose(texte: str) -> bool:
    """Le bloc porte-t-il autre chose qu'une phrase de « rien trouvé » ?

    On RETIRE les phrases sans contenu, puis on mesure le RESTE — au lieu de
    tester « contient une sentinelle ». La différence n'est pas stylistique :
    un bloc peut être vide ET annoté, et le test par contenance échouait
    exactement là (voir la note sur `_ANNOTATIONS_SANS_CONTENU`).
    """
    t = (texte or "").strip()
    for phrase in _SENTINELLES_VIDE:
        t = t.replace(phrase, " ")
    # ⚠️ On coupe de l'amorce jusqu'a la FIN DU PARAGRAPHE, pas jusqu'a la
    # premiere parenthese fermante. Mon premier motif faisait `\([^)]*\)` et il
    # etait faux : la note contient elle-meme des parentheses -- « resultat(s)
    # ecarte(s) » -- donc il s'arretait apres « resultat( » et laissait 44
    # caracteres de residu, assez pour repasser le plancher. Meme forme de
    # defaut que `<[^>]+>` qui avalait les seuils « < 6,5 % ».
    for amorce in _ANNOTATIONS_SANS_CONTENU:
        # La parenthèse ouvrante est OPTIONNELLE : les deux annotations
        # d'origine en ont une, la phrase de panne n'en a pas.
        t = re.sub(r"\(?\s*" + re.escape(amorce) + r".*?(?:" + chr(10)
                   + r"\s*" + chr(10) + r"|$)", " ", t, flags=re.S)
    return len(re.sub(r"\s+", " ", t).strip()) >= _MIN_CAR_BLOC_UTILE

def consignes_abstention(valeurs: dict) -> list:
    """Les consignes d'abstention à ajouter, vu le contenu réel des blocs.

    `valeurs` : {clé de bloc -> texte servi}. Pour chaque critère déclaré, si
    AUCUN de ses blocs sources ne porte quelque chose, on rend sa consigne.
    """
    out = []
    for critere, consigne in sorted(_ABSTENTION_PAR_CRITERE.items()):
        sources = blocs_du_critere(critere)
        if sources and not any(bloc_porte_quelque_chose(valeurs.get(c, ""))
                               for c in sources):
            out.append(consigne)
    return out

def blocs_du_critere(n: int) -> tuple:
    """Les clés des blocs qui servent le critère `n`.

    Dérivé de la déclaration, jamais réécrit : c'est le premier usage réel de
    `_BLOCS_PROMPT` au-delà de l'ordre de sacrifice, et il montre à quoi elle
    sert. `blocs_du_critere(1)` = les sources de la position du Gouvernement,
    donc du seul critère qui DISQUALIFIE.
    """
    return tuple(cle for cle, _e, _h, _rp, _prot, crits in _BLOCS_PROMPT
                 if n in crits)

# Message `system` de la réponse parlementaire : rôle, registre, garde-fous
# d'exactitude et glossaire — partie stable, réutilisée à chaque appel.
SYSTEME_REPONSE_PARLEMENTAIRE = """Vous êtes rédacteur au sein d'un cabinet ministériel. Vous rédigez le projet de réponse à une question écrite (QE) d'un parlementaire, portant sur la sphère sociale et médico-sociale française. Le texte sera publié au Journal officiel.

REGISTRE
- Style administratif, formel, factuel, à la troisième personne (« Le Gouvernement… », « les services de l'État… »). Jamais de première personne, jamais de formule commerciale ni de politesse.
- Prose continue uniquement : aucun titre, aucune puce, aucune liste, aucune numérotation. Les éléments multiples s'enchaînent en phrases complètes reliées par des connecteurs (« par ailleurs », « en outre », « à cet égard »).
- Aucune redondance : ne répétez pas une idée déjà exprimée.
- Le texte rendu est définitif : aucun crochet, aucun emplacement à compléter (« [montant] », « [à préciser] », « [mesures existantes] ») ne doit y subsister. Les crochets des consignes désignent un élément à remplacer par un fait tiré des contextes fournis ; s'il manque, la phrase est reformulée sans lui.
- Terminez par une phrase de clôture réaffirmant l'engagement du Gouvernement, sans élargir à des sujets éloignés.

EXACTITUDE (impératif — une erreur factuelle disqualifie la réponse)
- Ne citez un chiffre, un montant, un effectif, un pourcentage ou une date d'entrée en vigueur QUE s'il figure explicitement dans l'un des contextes fournis autres que le contexte parlementaire (voir ci-dessous). À défaut, restez qualitatif (« un montant revalorisé chaque année », « plusieurs centaines de structures ») : n'avancez jamais une valeur non sourcée, même plausible.
- N'écrivez jamais « loi n° AAAA-NNN », « décret n° AAAA-NNN », ni une référence d'article précise, si ce numéro exact ne figure pas dans un contexte fourni. Désignez alors le texte par son objet (« la loi relative à… », « le décret encadrant… », « un décret d'application est attendu »).
- Ne développez jamais un sigle absent des contextes fournis et du glossaire ci-dessous ; dans le doute, conservez le sigle seul.
- N'affirmez un lien juridique (« l'article X impose Y ») que si le texte fourni l'énonce clairement.
- La portée d'un article est bornée par sa position dans le code, indiquée après son numéro (livre, titre, chapitre, section : profession, public ou type de locaux concernés). Ne l'étendez jamais au-delà : un article du chapitre des masseurs-kinésithérapeutes ne dit rien des autres professions de santé ; un article sur les locaux d'habitation ne dit rien des établissements recevant du public. Si aucun texte fourni ne recoupe le sujet de la question, n'en citez aucun et renvoyez au cadre général.
- En cas de valeurs contradictoires pour une même donnée, retenez celle de la source la plus récente et précisez sa date.
- Le CONTEXTE PARLEMENTAIRE (réponses ministérielles antérieures) sert d'abord au registre, au ton et aux axes d'argumentation. Par défaut, n'en reprenez aucun chiffre, montant, effectif, pourcentage, date d'entrée en vigueur ni numéro de loi, de décret ou d'article : ces éléments y sont datés, donc présumés périmés — ils doivent alors provenir des TEXTES JURIDIQUES APPLICABLES, des DOCUMENTS DE RÉFÉRENCE ou de la RECHERCHE INTERNET.
- EXCEPTION : lorsqu'une réponse ministérielle est marquée « TRAME FACTUELLE ADMISSIBLE » dans le contexte (proximité élevée, récente, même législature) et qu'elle traite le même texte ou le même dispositif que la question, vous pouvez en reprendre l'enchaînement des faits — succession des textes, décisions de justice, dates-clés, échéances — à trois conditions : (1) recouper chaque élément avec les TEXTES JURIDIQUES, les DOCUMENTS DE RÉFÉRENCE ou la RECHERCHE INTERNET ; (2) ne rien reprendre qu'une source plus récente contredit ; (3) ne pas présenter comme actuel un chiffre ou un état du droit que rien de récent ne confirme. Une réponse marquée « registre seulement » (ancienne ou autre législature) reste cantonnée au ton et aux axes.
- Un texte (loi, décret, arrêté) que la RECHERCHE INTERNET ou un DOCUMENT DE RÉFÉRENCE donne comme **publié** — date de parution, référence au Journal officiel, numéro attribué, entrée marquée « TEXTE PUBLIÉ » — est publié : n'écrivez pas qu'il est « en préparation », « à venir », « à l'étude » ou que « le Gouvernement travaille à son élaboration ».
- Privilégiez systématiquement les textes et la stratégie les plus récents.
- Si un document de référence est une « fiche de référence » (chiffres-clés datés, texte récent), c'est la source à privilégier pour tout chiffre, date, numéro de texte ou définition de sigle : une valeur de fiche remplace toute valeur différente trouvée ailleurs — y compris dans le contexte parlementaire ou un rapport — alors réputée périmée. Si une fiche indique de ne pas citer une valeur, ne la citez pas.

GLOSSAIRE (n'introduire ces notions que si elles figurent dans la question ou un contexte)
AJPA = allocation journalière du proche aidant ; APA = allocation personnalisée d'autonomie ; AVA = assurance vieillesse des aidants ; PCH = prestation de compensation du handicap ; MDPH = maison départementale des personnes handicapées ; MDA = maison départementale de l'autonomie ; CNSA = Caisse nationale de solidarité pour l'autonomie ; PFR = plateforme d'accompagnement et de répit ; GIR = groupe iso-ressources ; CMI = carte mobilité inclusion ; RQTH = reconnaissance de la qualité de travailleur handicapé ; ESMS = établissements et services sociaux et médico-sociaux ; IGAS = Inspection générale des affaires sociales ; DREES = direction de la recherche, des études, de l'évaluation et des statistiques."""

_MOIS_FR = ("", "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre")

_DETAIL_JURIDIQUE = {
    1: "Aucune référence juridique n'est requise.",
    2: "Une seule phrase mentionne le cadre juridique général (ex. « Conformément au code de la sécurité sociale, … »).",
    3: "Un paragraphe court (2-3 phrases) expose le cadre juridique applicable, en citant un article clé si pertinent.",
    4: "Un paragraphe (4-5 phrases) analyse les implications juridiques, en citant explicitement deux à trois articles ou principes.",
    5: "Une analyse complète (un à deux paragraphes) cite précisément tous les articles pertinents, leurs interactions et leurs implications concrètes pour la question posée.",
}

_LONGUEUR_MOTS = {
    "Courte": "250 à 350 mots",
    "Moyenne": "450 à 600 mots",
    "Longue": "900 à 1200 mots",
}

# =====================================================================
# Vivier d'ouvertures, tiré au sort par question — INERTE (`VIVIER_OUVERTURES`)
# ---------------------------------------------------------------------
# Pourquoi. Le 08/09, retirer les phrases d'ouverture citées en exemple a rendu
# le modèle PLUS rigide, pas moins : 1 seule formule sur 7 cas, contre 2 pour le
# prompt qu'il remplaçait. L'hypothèse qui en sort — et que ce réglage teste —
# est que **les exemples n'étaient pas la cause de la rigidité mais sa seule
# protection** : privé de choix, le modèle retombe sur son défaut propre, unique.
#
# Ce qui varie est donc le GESTE — par quoi la réponse entre dans le dossier —
# et non la formulation : vingt façons de dire « le sujet est important » sont
# un seul geste, et un comptage par formule y verrait un faux succès.
#
# Le vivier est un fichier du dépôt, PAS du RAG : rien à retrouver par
# proximité de sens, seulement à tirer. L'utilisateur l'édite sans réingérer.
# ⚠️ DÉPLOIEMENT : `ouvertures_gestes.md` doit être poussé vers `QE_app` À CÔTÉ
#    d'`app.py`. S'il manque, le tirage est vide et le prompt revient à la
#    version héritée — dégradation silencieuse, d'où le contrôle au smoke.
# =====================================================================
VIVIER_OUVERTURES = False
GRAINE_OUVERTURES: Optional[int] = None   # None = tirage libre ; un entier = reproductible
GESTES_TIRES = 3
NOM_FICHIER_OUVERTURES = "ouvertures_gestes.md"

def _resoudre_fichier_ouvertures(depart: Optional[str] = None,
                                 niveaux: int = 3) -> str:
    """Trouve `ouvertures_gestes.md` en REMONTANT depuis `depart`.

    ⚠️ POURQUOI CE N'EST PLUS UN SIMPLE `join(dirname(__file__), …)`.
    L'ancienne forme résolvait le chemin À CÔTÉ DU MODULE QUI L'ÉCRIT. Tant que
    ce module est `app.py`, à la racine, c'est juste. Mais le constructeur de
    prompt doit être EXTRAIT dans un module partagé (Coordination, 13/09), et
    `tirer_gestes_ouverture` part avec lui : une fois dans un sous-paquet,
    `__file__` désigne le sous-répertoire, le fichier n'y est pas, et
    `charger_vivier_ouvertures` rend `{}` — donc **un tirage vide, sans aucun
    message**. Rien ne casse, la réponse s'appauvrit. C'est la panne que
    `ouvertures_gestes.md` a déjà produite le 11/09 en étant absent du
    déploiement, et elle serait revenue par un autre chemin.

    La remontée rend la résolution indépendante de l'endroit où vit le module.
    Elle ne remplace pas la déclaration de déploiement
    (`scripts/fichiers_du_deploiement.py`) : si le fichier n'est nulle part, il
    n'y a rien à remonter. Les deux servent deux absences différentes.

    Rend le premier chemin existant ; à défaut, le chemin canonique — pour que
    le message d'erreur nomme l'endroit attendu, et non le dernier essayé.
    """
    base = depart or os.path.dirname(os.path.abspath(__file__))
    canonique = os.path.join(base, NOM_FICHIER_OUVERTURES)
    courant = base
    for _ in range(max(0, niveaux) + 1):
        essai = os.path.join(courant, NOM_FICHIER_OUVERTURES)
        if os.path.exists(essai):
            return essai
        parent = os.path.dirname(courant)
        if parent == courant:
            break
        courant = parent
    return canonique

FICHIER_OUVERTURES = _resoudre_fichier_ouvertures()

# Les amorces héritées, retirées SEULEMENT quand le vivier est actif : elles
# font toutes le même geste, et le laisser en tête privilégierait ce geste-là.
# Chaînes exactes — un contrôle vérifie qu'elles sont bien présentes à l'inerte.
_AMORCES_HERITEES = (
    "Commencez par **souligner l'importance du sujet** pour le Gouvernement, sans reprendre les termes critiques du parlementaire. "
    "Utilisez des formulations comme : "
    "'Ce sujet est une priorité pour le Gouvernement, comme en témoignent [mesures existantes]', "
    "'Le Gouvernement est pleinement conscient des enjeux liés à [thème]', "
    "'Cette question, essentielle pour [public concerné], fait l'objet d'une attention constante de la part des services de l'État'. ",
    "1. **Reconnaissez l'importance du sujet** (sans valider les critiques) : "
    "'La question que vous soulevez touche à un enjeu majeur pour [public concerné], auquel le Gouvernement apporte une réponse structurée.' ",
    "Utilisez des formulations comme : "
    "'Votre proposition s'inscrit dans une dynamique que le Gouvernement partage, comme en attestent [mesures existantes].' "
    "'Nous partageons votre préoccupation pour [enjeu], et nos actions vont dans le sens de [objectif], comme le montre [exemple].' ",
)

def charger_vivier_ouvertures(chemin: Optional[str] = None) -> Dict[str, List[str]]:
    """Lit le vivier. Une ligne `# ` ouvre un geste, les `- ` sont ses exemples.

    Un geste sans exemple est ignoré : c'est ce qui laisse écrire de la prose
    dans le fichier sans qu'elle soit prise pour un geste. Fichier absent ou
    illisible -> dictionnaire vide, jamais d'exception.
    """
    try:
        with open(chemin or FICHIER_OUVERTURES, encoding="utf-8") as f:
            lignes = f.read().splitlines()
    except OSError:
        return {}
    vivier: Dict[str, List[str]] = {}
    courant = None
    for ligne in lignes:
        if ligne.startswith("# "):
            courant = ligne[2:].strip()
            vivier.setdefault(courant, [])
        elif ligne.startswith("- ") and courant:
            ex = ligne[2:].strip()
            if ex:
                vivier[courant].append(ex)
    return {g: ex for g, ex in vivier.items() if ex}

def tirer_gestes_ouverture(question: str, n: int = GESTES_TIRES,
                           graine: Optional[int] = None,
                           vivier: Optional[Dict[str, List[str]]] = None) -> str:
    """Tire `n` gestes distincts et un exemple de chacun. Bloc vide si pas de vivier.

    Reproductibilité : avec une graine (argument ou `GRAINE_OUVERTURES`), le
    tirage est **stable pour une question donnée** et varie d'une question à
    l'autre — c'est ce qu'il faut pour que `feat/eval` compare deux bras sans
    que le hasard s'ajoute à l'écart mesuré. Sans graine, tirage libre.
    """
    v = charger_vivier_ouvertures() if vivier is None else vivier
    if not v:
        return ""
    g = GRAINE_OUVERTURES if graine is None else graine
    rng = random.Random(f"{g}|{question}") if g is not None else random.Random()
    gestes = rng.sample(sorted(v), k=min(n, len(v)))
    lignes = [
        "- Ouverture — entrez dans le dossier par l'un des gestes ci-dessous, "
        "celui que CETTE question appelle. Les exemples montrent à quoi "
        "ressemble chaque entrée ; ils portent sur d'autres sujets et ne sont "
        "pas des phrases à reprendre."
    ]
    for geste in gestes:
        lignes.append(f"  • {geste} — par exemple : « {rng.choice(v[geste])} »")
    return "\n".join(lignes)

# Construit le message `user` de l'appel à Mistral (le `system` est
# SYSTEME_REPONSE_PARLEMENTAIRE, ajouté par call_mistral_parliamentary_response).
def build_parlementary_response_prompt(
    question: str,
    parliamentary_context: str,
    legal_context: str,  # Chaîne de caractères déjà triée et tronquée
    uploaded_documents: str,
    detail_juridique: int,
    longueur: str,
    response_orientation: str,
    custom_instructions: str,
    search_context: str,
    subquestions: Optional[List[str]] = None,
    consignes_sous_questions: Optional[Dict[str, str]] = None,
    # Bloc DÉJÀ FORMATÉ (par `format_positions_orales`), pas une liste de points :
    # il doit pouvoir entrer dans le dictionnaire que `reduire_contextes_pour_tenir`
    # allège, comme les quatre autres contextes.
    positions_orales: str = "",
) -> str:
    """
    Construit le message `user` : question, contexte récupéré (parlementaire,
    juridique, documentaire, web), demandes à traiter, et consignes propres à
    la demande (orientation, détail juridique, longueur, instructions libres).
    """
    # Mapping des orientations de réponse
    orientation_mapping = {
        "Répondre de façon neutre":
            "Adoptez un ton neutre et factuel. "
            "Commencez par **souligner l'importance du sujet** pour le Gouvernement, sans reprendre les termes critiques du parlementaire. "
            "Utilisez des formulations comme : "
            "'Ce sujet est une priorité pour le Gouvernement, comme en témoignent [mesures existantes]', "
            "'Le Gouvernement est pleinement conscient des enjeux liés à [thème]', "
            "'Cette question, essentielle pour [public concerné], fait l'objet d'une attention constante de la part des services de l'État'. "
            "Évitez absolument les formulations du type : 'comme vous le soulignez à juste titre', 'vous avez raison de pointer', ou 'la situation est effectivement préoccupante'. "
            "Privilégiez les faits, les chiffres, et les actions en cours.",

        "Répondre négativement aux propositions du parlementaire":
            "Répondez de manière **polie mais ferme**, en **recentrant le débat sur les actions du Gouvernement** plutôt que sur les critiques. "
            "Structurez votre réponse ainsi : "
            "1. **Reconnaissez l'importance du sujet** (sans valider les critiques) : "
            "'La question que vous soulevez touche à un enjeu majeur pour [public concerné], auquel le Gouvernement apporte une réponse structurée.' "
            "2. **Rappelez le cadre existant** : "
            "'Conformément à [texte juridique ou politique publique], les actions menées visent à [objectif].' "
            "3. **Expliquez les contraintes** (si nécessaire) : "
            "'Les marges de manœuvre sont encadrées par [contrainte légale/budgétaire], mais le Gouvernement agit dans le respect de ces règles pour [objectif].' "
            "4. **Mettez en avant les alternatives ou mesures en cours** : "
            "'Plutôt que [proposition du parlementaire], le Gouvernement a choisi de [mesure alternative], qui permet de [bénéfice].' "
            "Exemple : 'Plutôt qu'une refonte complète du dispositif, nous avons renforcé [mesure X], qui a déjà permis [résultat].' "
            "Évitez les formulations défensives comme 'nous ne pouvons pas' – préférez 'notre approche privilégie [solution], car [raison].'",

        "Répondre positivement aux propositions du parlementaire":
            "Saluez l'intérêt de la proposition **sans reprendre les critiques sous-jacentes**. "
            "Utilisez des formulations comme : "
            "'Votre proposition s'inscrit dans une dynamique que le Gouvernement partage, comme en attestent [mesures existantes].' "
            "'Nous partageons votre préoccupation pour [enjeu], et nos actions vont dans le sens de [objectif], comme le montre [exemple].' "
            "Évitez : 'Vous avez raison de souligner que...' → préférez : 'Votre attention à ce sujet rejoint nos priorités, illustrées par [action].'",

        "Répondre de manière technique et détaillée":
            "Fournissez une réponse **factuelle et technique**, en évitant tout commentaire sur les critiques du parlementaire. "
            "Structurez ainsi : "
            "1. **Cadre juridique** : 'Le dispositif actuel, défini par [article X], repose sur [principe].' "
            "2. **Données chiffrées** : 'Les derniers chiffres (source : [DREES/INSEE/...], [année]) montrent que [tendance].' "
            "3. **Mesures en cours** : 'Pour répondre à ces enjeux, [mesure A] et [mesure B] ont été mises en place, avec [résultat].' "
            "Utilisez un vocabulaire neutre et des verbes d'action : 'le Gouvernement a engagé', 'les services travaillent à', 'les résultats montrent que'."
    }

    now = datetime.now(pytz.timezone("Europe/Paris"))
    date_du_jour = f"{now.day} {_MOIS_FR[now.month]} {now.year}"

    # Vivier actif : on retire les amorces héritées (toutes le même geste) et on
    # injecte les gestes tirés. Inerte : le prompt est celui de la production,
    # aux octets près -- c'est le contrôle du smoke.
    orientation_txt = orientation_mapping.get(response_orientation, "")
    bloc_ouvertures = ""
    if VIVIER_OUVERTURES:
        tire = tirer_gestes_ouverture(question)
        if tire:
            for _amorce in _AMORCES_HERITEES:
                orientation_txt = orientation_txt.replace(_amorce, "")
            bloc_ouvertures = chr(10) + tire

    # Cinquième bloc, POSITIONS EXPRIMÉES EN SÉANCE. Inerte : la chaîne est
    # VIDE, donc le prompt est celui de la production aux octets près -- c'est
    # ce que vérifie le témoin négatif du smoke, pas une relecture.
    bloc_orales = ""
    if POSITIONS_ORALES:
        _entrees = (positions_orales or "").strip()
        if _entrees:
            bloc_orales = (
                chr(10) + chr(10)
                + "POSITIONS EXPRIMÉES EN SÉANCE — ce que le Gouvernement a dit "
                  "oralement, daté et attribué à la personne qui l'a dit. "
                  "À CITER, pas à imiter : le registre oral n'est pas celui "
                  "d'une réponse écrite. N'attribuez la position qu'à son "
                  "orateur, jamais à un ministère."
                + chr(10) + _entrees
            )

    # C1 — ABSTENTION. Inerte : la chaîne est VIDE, donc le prompt est celui de
    # la production aux octets près. Les valeurs viennent des paramètres du
    # constructeur ; la liste des blocs à consulter est DÉRIVÉE de la
    # déclaration (`blocs_du_critere`), jamais réécrite ici.
    bloc_abstention = ""
    if ABSTENTION_POSITION:
        _valeurs_blocs = {
            "parliamentary_context": parliamentary_context,
            "legal_context": legal_context,
            "uploaded_documents": uploaded_documents,
            "search_context": search_context,
            "positions_orales": positions_orales,
        }
        _cons = consignes_abstention(_valeurs_blocs)
        if _cons:
            bloc_abstention = "".join(chr(10) + "- " + c for c in _cons)

    detail = detail_juridique if detail_juridique in _DETAIL_JURIDIQUE else 3
    detail_consigne = _DETAIL_JURIDIQUE[detail]
    if detail >= 2:
        detail_consigne += (" Ne citez le numéro d'un article que s'il figure dans les "
                            "textes juridiques fournis ci-dessus ; sinon, renvoyez au code "
                            "concerné sans numéro.")

    mots_cible = next((v for k, v in _LONGUEUR_MOTS.items() if longueur.startswith(k)),
                      "450 à 600 mots")

    if subquestions:
        demandes = "\n".join(f"{i}. {sq}" for i, sq in enumerate(subquestions, 1))
    else:
        demandes = "Déduire les demandes de la question elle-même."

    # Consignes saisies par le rédacteur pour une sous-question précise. Elles
    # peuvent porter une information NON PUBLIQUE (décret en cours de rédaction,
    # effectif non encore publié) : c'est le canal prévu pour ça, et l'outil ne
    # peut pas la connaître autrement. On les donne donc comme des éléments
    # établis — mais on borne cette autorité au point visé, pour qu'elle ne
    # serve pas de laissez-passer général contre les règles d'exactitude.
    if consignes_sous_questions:
        _lignes = "\n".join(f"- Pour « {sq} » : {c}"
                             for sq, c in consignes_sous_questions.items())
        consignes_line = (
            "\nCONSIGNES DU RÉDACTEUR, PAR SOUS-QUESTION — elles émanent de "
            "l'administration et peuvent porter une information non encore "
            "publique. Tenez-les pour établies et suivez-les sur le point "
            "qu'elles visent ; elles ne dispensent d'aucune règle d'exactitude "
            "pour le reste de la réponse.\n" + _lignes + "\n")
    else:
        consignes_line = ""

    custom_line = (
        f"- Instruction spécifique impérative : {custom_instructions}\n"
        if custom_instructions else ""
    )

    prompt = f"""Nous sommes le {date_du_jour}.

QUESTION DU PARLEMENTAIRE
{question}

DEMANDES À TRAITER — traitez chacune, dans cet ordre. Si le contexte fourni ne permet pas de répondre précisément à une demande, indiquez-le (renvoi au cadre général, mesure à l'étude, temporisation) — n'avancez jamais un chiffre, une date ou un texte non étayé pour combler.
{demandes}
{consignes_line}
CONTEXTE PARLEMENTAIRE — réponses ministérielles antérieures à des questions proches. Registre, ton et axes d'argumentation. Une entrée marquée « TRAME FACTUELLE ADMISSIBLE » (proximité élevée, récente, même législature) et portant sur le même texte / dispositif que la question peut fournir l'enchaînement des faits, à recouper avec les autres contextes ; une entrée « registre seulement » n'apporte que le ton.
{parliamentary_context}

TEXTES JURIDIQUES APPLICABLES — prioritaires en cas de contradiction avec une autre source. Chaque article est situé dans son code (livre, titre, chapitre, section) : cette position borne sa portée. Un article qui ne recoupe pas le sujet de la question n'est pas à citer.
{legal_context}

DOCUMENTS DE RÉFÉRENCE
{uploaded_documents}

RECHERCHE INTERNET — actualités et positions du Gouvernement. Chaque résultat est daté et sourcé. Une entrée marquée « TEXTE PUBLIÉ » atteste que le texte visé est paru : tenez-le pour publié (et applicable à sa date d'entrée en vigueur), jamais pour « à venir » ou « en préparation ».
{search_context}{bloc_orales}

CONSIGNES DE RÉDACTION
- Orientation : {orientation_txt}{bloc_ouvertures}{bloc_abstention}
- Les crochets [ … ] des tournures ci-dessus sont des emplacements : remplacez-les par un élément des contextes fournis, ou reformulez la phrase sans eux. Aucun crochet ne doit subsister dans la réponse.
- Avant de rédiger, repérez dans les contextes fournis les éléments qui répondent DIRECTEMENT à la question : texte applicable et sa succession, décision de justice, échéance, chiffre daté, position récente du Gouvernement. Construisez la réponse dessus. Ne restez pas au niveau général quand un contexte contient une réponse précise ; à l'inverse, si aucun contexte ne traite frontalement une demande, dites-le (sujet à l'étude, non tranché) sans meubler.
- Corps de la réponse : rappel du cadre juridique et des chiffres disponibles, puis mesures en cours en privilégiant les plus récentes et l'année budgétaire courante. Intégrez les éléments des documents de référence puis de la recherche internet sans nommer la source.
- N'annoncez pas d'échéance déjà passée à la date du jour.
- Si une proposition du parlementaire est pertinente, indiquez qu'elle sera étudiée.
- Niveau de détail juridique : {detail}/5. {detail_consigne}
- Longueur cible : {mots_cible}. Si le sujet ne tient pas dans cette limite, concentrez-vous sur l'enjeu, le cadre juridique et la conclusion. Ne rendez jamais une réponse qui s'achève sur une phrase tronquée.
{custom_line}
PROJET DE RÉPONSE :"""
    return prompt


# =============================================================================
# LE PROFIL DE PRODUCTION -- les valeurs APPLIQUEES, pas les defauts declares
# =============================================================================
# Coordination (13/09) : « un module partage garantit le meme gabarit, pas les
# memes valeurs. Sans ca, les deux copies du gabarit disparaissent, les deux
# copies des reglages restent. »
#
# ⚠️ LA SIGNATURE DE `generate_response` MENT SUR DEUX DE CES VALEURS. Elle
# declare `detail_juridique: int = 1` et `model_size: str = "small"`, mais ces
# defauts NE S'APPLIQUENT JAMAIS : le site d'appel passe les neuf parametres
# explicitement, tous depuis des widgets. Le vrai defaut est celui du WIDGET.
#
#   detail_juridique   signature 1       -> production 3      (slider value=3)
#   model_size         signature "small" -> production "large" (MODELE_MISTRAL_DEFAUT)
#
# `model_size` compte double : il choisit la ligne de TOKEN_LIMITS (donc
# 1500/1500/3000/5000 et non 1000/1500/1500/2000) ET la fenetre du modele.
#
# ⚠️ `subquestions` vaut None : le decoupage en sous-questions N'EST PAS
# automatique, il faut cliquer « Decomposer en sous-questions ». L'en-tete
# « DEMANDES A TRAITER -- traitez chacune, dans cet ordre » reste pourtant, suivi
# de « Deduire les demandes de la question elle-meme » : le modele recoit l'ordre
# d'ENUMERER des demandes qu'il doit LUI-MEME inventer. Un harnais qui passerait
# deux sous-questions ne mesurerait donc PAS le cas nominal.
PROFIL_PRODUCTION: Dict[str, Any] = {
    "longueur": "Courte (300 mots)",        # selectbox index=0 -> « 250 a 350 mots »
    "response_orientation": "Repondre de facon neutre",   # selectbox index=0, 1er de 4
    "detail_juridique": 3,                 # slider value=3 -- PAS le 1 de la signature
    "custom_instructions": "",
    "subquestions": None,                  # pas de decoupage par defaut
    "consignes_sous_questions": None,      # initialise a {} dans l'interface
    "positions_orales": "",
}

# Ce qui n'est pas un parametre du constructeur mais gouverne l'assemblage en
# amont -- un harnais qui reproduit les blocs en a besoin, et c'est par la que
# l'ecart `small`/`large` est passe.
PROFIL_PRODUCTION_AMONT: Dict[str, Any] = {
    "model_size": "large",                 # PAS le "small" de la signature
    "include_legal_articles": True,        # toggle value=True
    "max_legal_articles": 3,               # = detail_juridique
    "must_contain": "",
}

# ⚠️ L'orientation porte une apostrophe typographique dans le code de
# production : cette valeur est celle que l'interface envoie, a l'octet. Elle est
# reconstruite depuis le texte reel plutot que retapee, pour que la comparaison
# de parite ne tombe pas sur un caractere invisible.
ORIENTATIONS = (
    "Répondre de façon neutre",
    "Répondre négativement aux propositions du parlementaire",
    "Répondre positivement aux propositions du parlementaire",
    "Répondre de manière technique et détaillée",
)
PROFIL_PRODUCTION["response_orientation"] = ORIENTATIONS[0]
