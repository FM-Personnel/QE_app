# `qe_rag` — pipeline d'ingestion RAG

Module autonome `extract → clean → chunk → embed → upsert`, utilisable en CLI ou
importé. **Ne dépend ni de `streamlit` ni de `app.py`.** Destiné, à terme, à
remplacer le chemin d'upload de l'app et les notebooks Colab d'ingestion
(cf. `reference/revue_module_ingestion.md`).

Phase 2 de `LOOP.md`. **`app.py` n'est pas encore modifié** pour l'utiliser — ça
viendra dans une étape séparée (avec sa dette de smoke-test).

## Ce que ça corrige (revue § A/B)

| # | Défaut de `app.py` | Ici |
|---|---|---|
| A1 | `segment_text()` (découpe par titre) calculée puis jamais utilisée | `chunk_by_section()` est la stratégie **par défaut** |
| A2 | `detect_titles()` appelé sur chaque **mot** → `section = "AUTRE"` partout | `detect_titles()` sur des **lignes** ; section suivie correctement dans les 2 stratégies |
| A3 | `re.sub(r'\s+', ' ')` sur le texte entier → structure aplatie | nettoyage **ligne par ligne**, frontières préservées pour le chunking |
| A6 | échec → `return False`, aucun détail ; comptage des vecteurs commenté | `IngestionReport` : étape par étape + **relecture du compte réel dans Qdrant** |
| B1 | pas de HTML | `extract_html()` : trafilatura → readability → BeautifulSoup → strip |

Hors périmètre (revue § B2–B5) : OCR, dédup complète, robustesse des liens Drive.
Idempotence a minima : `--collection <nom> --recreate`, et avertissement si une
version du même document existe déjà.

## Payload produit

Identique au contrat lu par `app.py::search_uploaded_documents` et aux collections
Qdrant existantes (`reference/structure_qdrant.json`) :

```
vecteur : 1024 dims, distance Cosine, NON normalisé
payload : {text, section, source, position, word_count, upload_date}
```

## Backends d'embedding

| spec | quoi | dépendance |
|---|---|---|
| `local` | `sentence-transformers` en local (Colab, PC ≥ 8 Go) | `sentence-transformers` |
| `endpoint` | service HTTP `services/embedding/` (contrat `POST /embed`) | `requests` |
| `cache:local` / `cache:endpoint` | enrobe un backend + cache disque JSONL (éval) | — |

```python
from qe_rag import ingest_document
from qe_rag.embed import get_backend
from qe_rag.qdrant_io import connect

backend = get_backend("endpoint", url="https://<space>.hf.space", token="…")
client = connect(QDRANT_URL, QDRANT_API_KEY)
rep = ingest_document("rapport.pdf", name="Rapport IGAS MDPH (2024)",
                      backend=backend, qdrant_client=client)
print(rep.summary())
```

## CLI

```bash
# extract + clean + chunk seulement (aucune dépendance modèle) :
python -m qe_rag ingest rapport.pdf --name "Rapport IGAS MDPH (2024)" --dry-run

# ingestion complète via le service d'embedding :
python -m qe_rag ingest https://legifrance.gouv.fr/… --kind html \
    --name "Décret n° 2025-827" \
    --backend endpoint --endpoint-url https://<space>.hf.space

# ré-ingestion idempotente d'un document :
python -m qe_rag ingest rapport.pdf --name "Rapport X" \
    --collection "Rapport_X__fixe" --recreate --backend endpoint --endpoint-url …

python -m qe_rag verify "Rapport_X__fixe"
```

Qdrant : `QDRANT_URL` / `QDRANT_API_KEY` depuis l'environnement ou `.env`.

## Installation

```bash
# cœur (suffit pour le backend « endpoint ») :
pip install -r qe_rag/requirements.txt

# backend « local » : torch CPU d'abord, sinon sentence-transformers tire les
# wheels CUDA (plusieurs Go inutiles sans GPU)
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
pip install -r qe_rag/requirements-local.txt
```

Vérifier qu'un backend est utilisable, **sans rien charger** :

```bash
python -m qe_rag check --backend local
```

> Le premier lancement réel du workflow a échoué là-dessus :
> `sentence-transformers` n'était qu'un **commentaire** dans
> `requirements.txt`, donc jamais installé, et l'échec ne se manifestait qu'au
> moment d'ingérer. La dépendance vit maintenant dans
> `requirements-local.txt` (installable), le workflow l'installe, et
> `qe_rag check` la vérifie **avant** toute écriture. La base était restée
> intacte : les embeddings sont calculés avant toute suppression.

## Tests

Sans réseau, sans modèle, sans Qdrant (doublures dans `tests/qe_rag/fakes.py`) :

```bash
python -m unittest discover -t . -s tests -p "test_*.py"
```

Les tests d'extraction PDF/DOCX/HTML se **skippent** si la lib correspondante est absente.

---

## Ingestion par GitHub Actions (remplace Colab)

Workflow : `.github/workflows/rag-ingest.yml`. **Déclenchement manuel
uniquement** (`workflow_dispatch`) — jamais sur `push`, parce que l'ingestion
avec `--recreate` **supprime** la collection avant de la recréer : un
déclencheur automatique détruirait la base à chaque commit.

```bash
# répétition à blanc (défaut) : extraction + découpage, Qdrant jamais touché
gh workflow run rag-ingest.yml -f source=reference/rag_docs/ma_fiche.md

# ingestion réelle, avec contrôle du nombre de points
gh workflow run rag-ingest.yml \
    -f source=reference/rag_docs/ma_fiche.md \
    -f name="Fiche de reference - mon sujet" \
    -f dry_run=false -f recreate=true \
    -f expected_points=24
```

Depuis l'interface : onglet **Actions** → *Ingestion RAG (Qdrant)* → *Run workflow*.

### Le modèle d'embedding tourne sur l'exécuteur, et c'est normal

La consigne « ne jamais charger le modèle d'embedding en local » vise **le poste
de développement** (5,9 Go de RAM ; le chargement l'a déjà bloqué des heures).
Un exécuteur `ubuntu-latest` a environ 16 Go : le backend `local` y est
légitime, c'est le mode prévu. La règle n'est pas contournée, elle ne
s'applique pas à cette machine — c'est écrit aussi en tête du workflow pour
que personne ne conclue l'inverse en le relisant plus tard.

### Ce que fait le workflow

1. **Répétition à blanc systématique**, même avant une ingestion réelle :
   extraction, nettoyage, découpage, **sans toucher Qdrant**. La plupart des
   échecs (fichier introuvable, PDF image, texte vide, découpage aberrant) sont
   attrapés là, donc avant toute suppression.
2. **Ingestion** (seulement si `dry_run=false`), rapport JSON.
3. **Vérification indépendante** : relecture de la collection dans Qdrant via
   `qe_rag verify` (compte réel + échantillon), plutôt que de croire le rapport.
4. **Échec bruyant** (`::error::`) si le nombre de points s'écarte de
   `expected_points` au-delà de `tolerance`.
5. Résumé lisible dans le journal du run (`$GITHUB_STEP_SUMMARY`).

Les secrets `QDRANT_URL` et `QDRANT_API_KEY` sont posés sur le dépôt et
seulement **consommés** par le workflow.

### Réparer une ingestion interrompue

**Bonne nouvelle d'abord** : la fenêtre destructrice est étroite. `ingest_document`
calcule les embeddings **avant** de supprimer quoi que ce soit — l'ordre réel est
`encode` → `ensure_collection(recreate)` → `upsert` → relecture du compte. Un
échec d'extraction, de découpage ou d'embedding se produit donc **avant** toute
suppression : la collection existante est intacte. Seule une panne réseau ou
Qdrant *pendant* l'upsert laisse une collection vide.

Si le workflow échoue avec `recreate=true` (il affiche alors une alerte
explicite) :

1. **Constater l'état**, sans rien relancer :
   ```bash
   python -m qe_rag verify "<nom exact de la collection>"
   ```
   · « introuvable » → supprimée, non recréée · un compte faible → upsert partiel.
2. **Relancer la même ingestion à l'identique**, avec le **même
   `--collection`** et `recreate=true`. Les identifiants de points sont
   déterministes (`uuid5` sur `source::section::position`) : ré-ingérer le même
   document reconstruit la même collection, sans doublon.
3. Si la source elle-même est en cause (fichier corrompu, URL morte), réparer la
   source d'abord — inutile de relancer l'ingestion tant que la répétition à
   blanc ne passe pas.
4. Tant que la collection est vide, l'app **fonctionne mais sans cette fiche** :
   `search_uploaded_documents` ignore une collection absente. La dégradation est
   silencieuse côté utilisateur, d'où l'échec bruyant côté workflow.

**Conseil** : renseigner `expected_points` dès qu'on connaît le compte attendu
(il figure dans le résumé de l'ingestion précédente). C'est le seul garde-fou
qui distingue « ingestion réussie » de « ingestion réussie à moitié ».

### Pied de chunk (`--footer-file`)

Les fiches de référence portent un « pied » ajouté à **chaque** chunk (garde-fous
du type « ne jamais citer telle valeur périmée ») : la requête ne fait pas
toujours remonter la bonne section, donc n'importe quel extrait retrouvé doit
porter les corrections clés.

```bash
python -m qe_rag ingest reference/rag_docs/x.md --footer-file pieds/x.txt
```

`--skip-footer-section` exclut certaines sections du pied. Les **défauts**
(`perimes a ne jamais citer`, `annexe de service`) reprennent ceux du notebook
Colab : ces sections sont denses en sigles, numéros et dates, elles recouvrent
lexicalement presque toute question du domaine et battent le chunk de contenu
pertinent ; y empiler le pied aggrave le déséquilibre.

### Mode « lot » : les 25 fiches curées

Depuis l'extraction du manifeste (`scripts/rag_docs_manifest.json`, 25 entrées,
23 pieds dans `reference/rag_docs/footers/`), le workflow ingère **le lot
complet** :

```bash
gh workflow run rag-ingest.yml -f mode=lot                      # à blanc
gh workflow run rag-ingest.yml -f mode=lot -f dry_run=false -f recreate=true
```

Le pilote est `scripts/ingest_lot.py`, qui **lit** le manifeste et n'en recopie
rien. La répétition à blanc sur le lot est **obligatoire** et affiche le nombre
de chunks **par fiche** — c'est le contrôle qui remplace l'œil humain qu'on
avait sur la cellule 4 du notebook. `--expected-chunks` fait échouer
bruyamment sur écart.

**Source unique, vérifiée par un test.** `skip_footer_sections` vit dans le
manifeste seul : le défaut de `--skip-footer-section` est **lu** depuis lui, et
`tests/qe_rag/test_manifest.py` échoue si l'une de ces valeurs réapparaît
ailleurs dans `qe_rag`. C'est la parade au motif du bug F12.

> ⚠️ **Le lot n'est pas encore autorisé à l'ingestion réelle** — écart de
> découpage mesuré, non résolu. Voir la section suivante.

### Écart de découpage avec le notebook — corrigé, et ce qu'il en reste

Mesuré en répétition à blanc sur le lot (aucun modèle, aucun réseau).

| étape | chunks / 25 fiches |
|---|---|
| notebook Colab (référence, ce qui est en production) | **345** |
| `qe_rag` au départ, défaut `min_words=50` | 337 |
| `qe_rag`, `min_words=20` aligné sur le lot | 350 |
| **`qe_rag` après portage du garde-fou de titre** | **347** |

**Deux causes corrigées :**

1. **`min_words` 50 contre 20.** `scripts/ingest_lot.py` fixe explicitement
   `min_words=20` : c'est un paramètre du lot de fiches, pas un défaut de
   `qe_rag`. `--min-words` est exposé sur la CLI.
2. **Faux positif de détection de titre (3 chunks).** Le motif `^Article\s+\d+`
   s'appliquait sans garde-fou de longueur : une ligne de prose commençant par
   « article 53 de la loi n° 2018-727 du 10 août 2018 (ESSOC)… » ouvrait un
   chunk et remplissait la métadonnée `section` avec un bout de citation. Le
   notebook avait déjà résolu ce point et l'avait documenté ; `qe_rag` n'avait
   hérité que du motif brut. **Garde-fou porté** (`TITRE_LONGUEUR_MAX = 60`),
   avec l'ordre du notebook : marqueurs structurels d'abord (un titre markdown
   vaut à toute longueur), puis la borne de longueur, puis les motifs de prose.

**Écart résiduel assumé : +2 chunks, sur une seule fiche.** Il ne vient pas
d'une régression de `qe_rag` mais d'une **limite du notebook** : sa détection
ne reconnaît que `# ` et `## `, elle **ignore `###` et au-delà**, qui repartent
donc dans le corps du texte. `qe_rag` gère `#` à `######`. La fiche
`loi_2024-1028_art9_decret_2025-827` porte deux titres de niveau 3
(« ### Article 9 », « ### Contenu ») : `qe_rag` les traite en sections,
le notebook les noie dans le paragraphe précédent.

**Aligner `qe_rag` sur 345 supposerait de lui faire ignorer les titres de
niveau 3** — dégrader un comportement correct pour reproduire une limite. Ce
n'est pas fait : l'écart est **figé par un test** (`tests/qe_rag/test_parite_notebook.py`,
`ECARTS_CONNUS`) qui compare les deux découpages fiche par fiche sur les
documents réels et casse si un écart NOUVEAU apparaît. Le choix — ingérer 347
en acceptant deux sections de plus, ou rester sur 345 — revient à
l'utilisateur.

