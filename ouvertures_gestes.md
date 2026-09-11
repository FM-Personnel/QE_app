# Vivier d'ouvertures — organisé par GESTE

Ce fichier est lu par `app.py` au moment d'assembler le prompt. **Il n'est pas
dans le RAG** : rien n'y est retrouvé par proximité de sens, on y tire au sort.
Vous pouvez l'éditer librement — aucune réingestion, aucun redéploiement de
modèle. Il suffit qu'il soit présent à côté d'`app.py`.

## Ce qui est tiré, et pourquoi

**On tire des GESTES, pas des phrases.** Vingt manières de dire « le
Gouvernement prend ce sujet au sérieux » restent **un seul geste** : le
comptage afficherait une variété qui n'existe pas. Ce qui doit varier, c'est
**par quoi la réponse entre dans le dossier**.

Les exemples ne sont pas des phrases à recopier : ils sont là pour **montrer à
quoi ressemble** une ouverture sur un chiffre ou sur une échéance. Ils sont
volontairement attachés à un sujet précis, pour qu'aucun ne puisse être collé
tel quel dans une autre réponse.

## Comment éditer

Une ligne `# ` ouvre un geste, les lignes `- ` sont ses exemples. Tout le reste
est ignoré. Un geste sans exemple est ignoré. **Ajouter un geste, c'est ajouter
une manière d'ouvrir** — c'est ce qui a le plus d'effet ; ajouter un exemple à
un geste existant en a beaucoup moins.

---

# ouvrir sur le texte applicable

- L'article L. 245-1 du code de l'action sociale et des familles ouvre la prestation de compensation du handicap aux personnes dont les difficultés répondent aux critères fixés par décret.
- Le droit au répit des proches aidants est régi par l'article L. 14-10-1 du même code, dont le décret n° 2025-827 a précisé les conditions de mise en œuvre.
- La visite médicale de reprise après un arrêt de travail relève de l'article R. 4624-31 du code du travail, qui en fixe le délai et la charge.

# ouvrir sur un chiffre et sa date

- Au 31 décembre 2025, 1,3 million de personnes percevaient l'allocation aux adultes handicapés, soit une progression de 2,4 % sur un an.
- Les maisons départementales des personnes handicapées ont instruit 4,9 millions de demandes en 2024, pour un délai moyen de traitement de 4,4 mois.
- La dépense d'aide sociale départementale consacrée au grand âge s'est établie à 7,9 milliards d'euros en 2024.

# ouvrir sur une échéance

- Le décret d'application prévu par l'article 9 de la loi du 8 avril 2024 doit entrer en vigueur au 1er janvier 2027.
- La convention d'objectifs et de gestion signée avec la caisse nationale couvre la période 2023-2027 et fera l'objet d'un bilan à mi-parcours cette année.
- La généralisation du service public départemental de l'autonomie est prévue pour le 30 juin 2026.

# ouvrir sur une décision déjà prise

- Le Gouvernement a retenu, dans le cadre de la loi de financement de la sécurité sociale pour 2026, le relèvement du plafond de l'allocation journalière du proche aidant.
- La revalorisation exceptionnelle des salaires du secteur médico-social a été actée par l'avenant du 4 juin 2024, agréé le mois suivant.
- La création de 50 000 places de services à domicile a été arbitrée et figure dans la programmation pluriannuelle.

# ouvrir sur l'état d'un chantier

- Les travaux de refonte du système d'information des maisons départementales sont engagés depuis 2024 et leur déploiement se poursuit département par département.
- La concertation sur la réforme du financement des services d'aide à domicile s'est achevée en juin, et ses conclusions sont en cours d'arbitrage interministériel.
- L'expérimentation menée dans douze départements fait l'objet d'une évaluation dont les résultats sont attendus avant la fin de l'année.

# ouvrir en reformulant précisément ce que la question demande

- La question porte sur le sort des personnes dont l'accord de branche n'a pas été étendu : c'est le régime transitoire qui s'applique, et il mérite d'être détaillé.
- Deux situations doivent être distinguées : celle des travailleurs déjà admis en établissement et service d'accompagnement par le travail, et celle des personnes en attente d'une orientation.
- Ce qui est en cause n'est pas le principe du financement mais la clé de répartition entre l'État et les départements.

# ouvrir en soulignant l'importance du sujet

- L'accès aux soins visuels dans les territoires ruraux est un sujet sur lequel les délais constatés justifient une réponse détaillée.
- La protection des mineurs face aux contenus numériques mobilise plusieurs ministères et a fait l'objet de trois textes depuis 2023.
- La situation des aidants de personnes atteintes de troubles du neurodéveloppement appelle une attention particulière, que traduisent les mesures rappelées ci-après.
