import streamlit as st
import hashlib
import requests
import os
import pytz
import time
import unicodedata
import importlib.metadata
import re
import tempfile
import uuid
from pydantic import BaseModel, validator
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Union, Dict, Any, Literal, Set
from datetime import datetime, timedelta
from collections import defaultdict
from PyPDF2 import PdfReader
from docx import Document
from unstructured.partition.pdf import partition_pdf  # Optionnel, pour une extraction plus fine
from unstructured.partition.docx import partition_docx

try:
    import pdfplumber
    print("✅ pdfplumber est installé et fonctionnel.")
except ImportError:
    print("❌ pdfplumber n'est pas installé. Exécutez : pip install pdfplumber")
    raise

#################################################################
# 1. CLASSES, FONCTIONS UTILITAIRES, INITIALISATION DES VARIABLES
#################################################################

# --- Persistance de l'historique via session_state ---
def save_historique():
    # Limite le nombre d'entrées pour éviter la surcharge (ex: 50 dernières)
    if len(st.session_state.full_historique) > 50:
        # Supprime les entrées les plus anciennes
        sorted_entries = sorted(
            st.session_state.full_historique.items(),
            key=lambda x: x[1]["metadata"].get("timestamp", "")
        )
        st.session_state.full_historique = dict(sorted_entries[-50:])
    st.session_state.historique_cache = st.session_state.full_historique.copy()

def load_historique():
    if "historique_cache" in st.session_state:
        st.session_state.full_historique = st.session_state.historique_cache.copy()

# Pour permettre de renommer une collection
if 'show_rename_modal' not in st.session_state:
    st.session_state.show_rename_modal = False
if 'current_doc_to_rename' not in st.session_state:
    st.session_state.current_doc_to_rename = None

# Initialisation des états de certaines variables si non existants
if 'use_priority_docs' not in st.session_state:
    st.session_state.use_priority_docs = False
if 'priority_docs' not in st.session_state:
    st.session_state.priority_docs = []

# Estimation du nombre de tokens d'un texte en français
def estimate_tokens(text):
    """Estime le nombre de tokens pour Mistral Large (1 token ≈ 4 caractères en français)."""
    return len(text) // 4

# Fonction utilitaire pour tronquer le texte
def truncate_text(text: str, max_tokens: int = 500) -> str:
    """Tronque un texte à un nombre maximal de tokens (1 token ≈ 4 caractères)."""
    max_chars = max_tokens * 4
    return (text[:max_chars] + "...") if len(text) > max_chars else text

# Fonction de conversion des dates
def safe_parse_date(date_str: Optional[str]) -> datetime:
    """Convertit une date hétérogène en datetime, ou datetime.min si invalide."""
    if not date_str:
        return datetime.min
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.min

# Clé numérique
def sort_key(article_num: str):
    """
    Transforme un identifiant d'article (ex: 'D146-12-1') en tuple numérique
    pour un tri correct.
    """
    if not article_num:
        return ("",)
    prefix = article_num[0]
    parts = re.findall(r'\d+', article_num)
    nums = [int(p) for p in parts]
    return (prefix, *nums)

# Charger l'historique au démarrage
load_historique()

# --- Configuration de la page (doit être unique et en premier) ---
st.set_page_config(
    page_title="Parlement RAG",
    page_icon="🗳️",
    layout="wide"   # ← élargit toute la page
)

# Classe  CSS pour tous les messages de statut
st.markdown("""
<style>
    /* ===== STYLES GLOBAUX ===== */
    /* Messages de statut (utilisé par status_placeholder et prep_placeholder) */
    .status-message, .prep-message {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.5rem auto !important;
        padding: 0.7rem 1rem;
        text-align: center;
        font-size: 14px;
        color: #555;
        background-color: #f0f2f6;
        border-radius: 6px;
        max-width: 600px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Style spécifique pour les messages dans les onglets */
    .prep-message {
        min-height: 40px;
        font-size: 15px;
    }
    /* Conteneur principal */
    .stApp {
        display: flex;
        flex-direction: column;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Expanders et contenu */
    div[data-testid="stExpander"] {
        margin: 0 auto 0.5rem auto !important;
        max-width: 1000px;
        width: 100%;
    }
    div.streamlit-expanderHeader {
        padding: 0.3rem 0.6rem !important;
        font-size: 13px !important;
        background-color: #f0f2f6 !important;
        border-radius: 4px !important;
    }
    div.streamlit-expanderContent {
        padding: 0.5rem 0.8rem !important;
        text-align: justify;
    }
    /* Titres et séparateurs */
    h3 {
        text-align: center;
        margin: 0.4rem auto !important;
        color: #0066cc;
    }
    hr {
        margin: 0.3rem auto;
        width: 80%;
        border: none;
        border-top: 1px solid #ddd;
    }
    /* Suppression des marges inutiles */
    .stMarkdown > div {
        margin: 0 !important;
    }
    p, ul, ol {
        margin: 0.2rem 0 !important;
        padding: 0 !important;
    }
    /* Style pour les expanders de l'historique */
    div[data-testid="stExpander"] > details > summary {
        padding: 0.5rem 1rem !important;
        background-color: #f0f2f6 !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
    }
    /* Espacement entre les entrées */
    .history-entry {
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Chargement des variables d'environnement ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CX = os.getenv("GOOGLE_CX", "").strip()

QDRANT_COLLECTION = "QuestionParlementaire"

if not QDRANT_URL.startswith("https://"):
    raise RuntimeError("❌ QDRANT_URL doit commencer par https://")
if not QDRANT_API_KEY:
    raise RuntimeError("❌ QDRANT_API_KEY manquant. Vérifiez votre fichier .env")
if not MISTRAL_API_KEY:
    raise RuntimeError("❌ MISTRAL_API_KEY manquant. Vérifiez votre fichier .env")

print("QDRANT_URL:", QDRANT_URL)
print("QDRANT_API_KEY (début):", QDRANT_API_KEY[:10], "...")

# --- 3. Chargement du modèle ---
LOCAL_MODEL_PATH = "./models/camembert_finetuned_progressive"
HUB_MODEL_PATH = "Whisler/camembert_finetuned_progressive"

@st.cache_resource
def load_embedding_model():
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            model = SentenceTransformer(LOCAL_MODEL_PATH)
            print("✅ Modèle chargé en local.")
        else:
            model = SentenceTransformer(HUB_MODEL_PATH)
            print("✅ Modèle téléchargé depuis HuggingFace Hub.")
        test_embedding = model.encode("Test de chargement du modèle.")
        VECTOR_SIZE = len(test_embedding)
        print("Dimension des embeddings:", VECTOR_SIZE)
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {str(e)}")
        raise

embedding_model = load_embedding_model()

# --- 4. Connexion à Qdrant ---
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10.0,
        check_compatibility=False
    )
    collections = qdrant_client.get_collections()
    print(f"✅ Connexion réussie. Collections disponibles: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"❌ Erreur de connexion à Qdrant: {e}")
    raise


# --- 5. Modèles Pydantic ---

# Modele Pydantic pour les articles juridiques
class BaseLegislativeRef(BaseModel):
    uid: str
    collection: str

class RetrievedLegalDocument(BaseModel):
    # Identifiants
    chunk_id: str
    num: str
    titre: str
    
    # Contenu
    contenu: str
    article_complet: str
    
    # Contexte
    contexte_hierarchique: str
    collection: str
    
    # Hiérarchie optionnelle
    partie: Optional[str] = None
    livre: Optional[str] = None
    titre_structure: Optional[str] = None
    chapitre: Optional[str] = None
    section: Optional[str] = None
    sous_section: Optional[str] = None
    paragraphe: Optional[str] = None
    sous_paragraphe: Optional[str] = None
    
    # Références législatives
    base_legislative: Optional[List[BaseLegislativeRef]] = None
    
    # Score du retrieval
    score: Optional[float] = None

# Modele Pydantic pour les documents generiques
class GenericDocument(BaseModel):
    # Identifiants
    uid: Optional[str] = None
    
    # Contenu
    text: str
    title: Optional[str] = None
    part: Optional[str] = None   # ex. "Annexe 7", "partie 1"
    
    # Métadonnées
    source: Optional[str] = None
    date_document: Optional[str] = None
    type_document: Optional[str] = None
    
    # Score du retrieval
    score: Optional[float] = None

# Modele Pydantic pour les reponses RAG
class ResponseDocument(BaseModel):
    # Identifiants
    uid: str
    # Contenu
    question: str
    reponse: str
    # Métadonnées
    legislature: Optional[str] = None
    chambre: Optional[str] = None   # Assemblée ou Sénat (à ajouter si tu l’as dans tes données)
    rubrique: Optional[str] = None
    analyse: Optional[str] = None
    ministeres_attribues: Optional[List[str]] = None
    # Dates
    date_question: Optional[str] = None
    date_reponse: Optional[str] = None
    # Références juridiques éventuelles
    textes_juridiques: Optional[List[str]] = None
    # Score du retrieval
    score: Optional[float] = None


#################################################################
# -------------- 2. PRINCIPALES FONCTIONS -----------------------
#################################################################

# --- 2a. Fonctions d'upload, d'indexation et d'embedding

# Fonction pour extraire le texte d'un document pdf
def extract_text(pdf_path: str, max_pages: Optional[int] = None) -> str:
    """Extrait le texte d'un PDF, avec gestion des erreurs et des pages vides (VOTRE FONCTION)."""
    text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                page_text = page.extract_text() or ""
                if page_text.strip():  # Ignore les pages vides
                    text.append(page_text)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction du PDF: {e}")
        return ""
    return "\n".join(text)

# Fonction pour extraire le texte d'un document Word
def extract_text_from_docx(file_path: str, use_unstructured: bool = False) -> str:
    """Extrait le texte d'un fichier Word."""
    if use_unstructured:
        elements = partition_docx(file_path)
        text = "\n\n".join([str(el) for el in elements])
    else:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip()

# Fonction pour nettoyer les textes extraits
def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les lignes trop courtes, les numéros de page et les artefacts (VOTRE FONCTION AMÉLIORÉE)."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Ignorer les lignes trop courtes ou purement numériques
        if len(line) > 10 and not re.match(r'^\d+$', line.strip()):
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    # Supprimer les espaces multiples et sauts de ligne redondants
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Recolle les mots coupés
    text = re.sub(r'\b\d{1,3}\b(?=\s|$|\.|\,)', '', text)  # Supprime les nombres isolés
    text = re.sub(r'PLFSS\s*20\d{2}\s*-\s*Annexe\s*\d+', "", text, flags=re.IGNORECASE)
    text = re.sub(r"Source\s*:.*?(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[•○◘]\s*|\bPage\s+\d+\b", "", text)
    text = re.sub(r"\bhttps?://\S+|\[.*?\]", "", text)
    return text.strip()

# Fonction de suppression du sommaire
def remove_summary(text: str) -> str:
    """Supprime le sommaire et les tables des matières (VOTRE FONCTION)."""
    cleaned = re.sub(
        r"(?i)(SOMMAIRE|TABLE DES MATIÈRES).*?(?=PARTIE\s+\d+|ANNEXE\s+\d+|Article\s+\d+|$)",
        "",
        text,
        flags=re.DOTALL
    )
    return cleaned.strip() if cleaned.strip() else text

# Fonction de pré-traitement des titres
def preprocess_for_titles(text: str) -> str:
    """Insère des sauts de ligne avant chaque titre (VOTRE FONCTION OPTIMISÉE)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    processed_text = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if detect_titles(sentence):
            processed_text.append(f"\n{sentence}\n")
        else:
            processed_text.append(sentence)
    text = " ".join(processed_text)
    title_patterns = [
        r"(Article\s*\d*\s*[-–]?)", r"(ANNEXE\s+\d+)", r"(PARTIE\s+\d+)",
        r"(Fiches?\s+d’?évaluation\s+préalable)", r"(\d+\.\s+)", r"([IVXLCDM]+\.\s+)"
    ]
    for pattern in title_patterns:
        text = re.sub(pattern, r"\n\1\n", text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

# Fonction de détection des titres
def detect_titles(line: str) -> bool:
    """Détecte les titres de manière robuste."""
    line = line.strip()
    if not line:
        return False
    regex_patterns = [
        r"^Article\s+\d+\s*[–—-]", r"^ANNEXE\s+\d+", r"^PARTIE\s+\d+",
        r"^(TITRE|Chapitre|Section)\s+\d+", r"^[IVXLCDM]+\.\s+", r"^\d+(\.\d+)*\s+"
    ]
    if any(re.match(p, line, flags=re.IGNORECASE) for p in regex_patterns):
        return True
    title_keywords = [
        "Article ", "ANNEXE ", "PARTIE ", "Fiches d’évaluation préalable",
        "Synthèse", "Conclusion", "TITRE ", "Chapitre ", "Section ",
        "I. ", "II. ", "III. ", "1. ", "2. "
    ]
    return any(keyword.lower() in line.lower() for keyword in title_keywords)

# Fonction de segmentation du texte
def segment_text(text: str, max_words: int = 300, min_words: int = 50) -> List[Dict[str, str]]:
    """Découpe le texte en segments avec titres (VOTRE FONCTION ADAPTÉE POUR QDRANT)."""
    blocks = re.split(r'\n\n|\.\s+', text)
    segments = []
    current_title = "AUTRE"
    current_content = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if detect_titles(block):
            if current_content:
                seg_text = ' '.join(current_content)
                if len(seg_text.split()) >= min_words:
                    segments.append({"title": current_title, "text": seg_text})
                current_content = []
            current_title = block
        else:
            current_content.append(block)
    if current_content:
        seg_text = ' '.join(current_content)
        if len(seg_text.split()) >= min_words:
            segments.append({"title": current_title, "text": seg_text})
    return segments

# Fonction pour préparer les chunks
def prepare_chunks_fixed(text: str, file_name: str, chunk_size=350, overlap=50) -> List[Dict]:
    """
    Découpe le texte en chunks fixes (~350 mots ≈ 512 tokens) avec overlap,
    en conservant le dernier titre détecté comme métadonnée.
    """
    import uuid
    words = text.split()
    chunks = []
    start = 0
    current_title = "AUTRE"

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        # Met à jour le titre courant si un mot ressemble à un titre
        for w in chunk_words:
            if detect_titles(w):
                current_title = w

        if len(chunk_words) >= 50:  # filtre chunks trop courts
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "source": file_name,
                    "section": current_title,  # dernier titre détecté
                    "position": start,
                    "word_count": len(chunk_words),
                    "upload_date": datetime.now().isoformat()
                }
            })
        start += chunk_size - overlap  # avance avec recouvrement

    return chunks

# Fonction pour l'upload et l'indexation
def process_and_index_document(file_path: str, file_type: str, collection_name: str, progress_callback=None, **kwargs):
    """Traite et indexe un document dans SA PROPRE COLLECTION avec suivi de progression et gestion des timeouts."""
    try:
        # 1. Extraction du texte
        if file_type == "pdf":
            raw_text = extract_text(file_path)
        else:
            raw_text = extract_text_from_docx(file_path)

        if not raw_text:
            if progress_callback:
                progress_callback(0, 1, "Échec : extraction du texte")
            return False

        if progress_callback:
            progress_callback(5, 100, "Extraction du texte terminée")

        # 2. Nettoyage et segmentation
        cleaned_text = clean_text(raw_text)
        segments = segment_text(cleaned_text) or [{"title": "Document", "text": cleaned_text}]

        if progress_callback:
            progress_callback(10, 100, "Nettoyage et segmentation terminés")

        # 3. Préparation des chunks fixes avec overlap
        preprocessed_text = preprocess_for_titles(cleaned_text)
        chunks = prepare_chunks_fixed(
            preprocessed_text,
            file_name=collection_name,
            chunk_size=350,   # ≈ 512 tokens
            overlap=50        # recouvrement pour ne pas perdre de contexte
        )

        if progress_callback:
            progress_callback(30, 100, f"Préparation des chunks terminée ({len(chunks)} chunks)")

        # 4. Génération des embeddings (par petits lots)
        texts = [chunk["text"] for chunk in chunks]
        total_chunks = len(chunks)
        embeddings = []

        for i in range(0, len(texts), 5):  # Réduit à 5 chunks par lot pour éviter la surcharge
            batch_texts = texts[i:i+5]
            try:
                batch_embeddings = embedding_model.encode(batch_texts).tolist()
                embeddings.extend(batch_embeddings)
            except Exception as e:
                if progress_callback:
                    progress_callback(0, 100, f"Erreur génération embeddings: {str(e)}")
                return False

            if progress_callback:
                current_chunk = min(i + 5, total_chunks)
                progress_callback(30 + int(30 * current_chunk / total_chunks),
                                100,
                                f"Génération des embeddings : {current_chunk}/{total_chunks}")

        # 5. Indexation dans Qdrant (par petits lots avec réessais)
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(
                models.PointStruct(
                    id=chunk["id"],
                    vector=embedding,
                    payload={
                        "text": chunk["text"],   # ← ajoute le texte du chunk
                        **chunk["metadata"]      # ← conserve les métadonnées
                    }
                )
            )
            
        # Réduit la taille des batches et réessais
        batch_size = 10  # Réduit de 50 à 10
        max_retries = 2   # Nombre maximal de réessais
        retry_delay = 2   # Délai entre les réessais en secondes

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            retry_count = 0
            success = False

            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                try:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True,
                    )
                except Exception as e:
                    st.error(f"Erreur Qdrant batch {i//batch_size+1}: {e}")
                    st.write("Exemple point:", batch[0].vector[:10], "dim=", len(batch[0].vector))
                    return False

            while retry_count < max_retries and not success:
                try:
                    kwargs["qdrant_client"].upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True,
                    )
                    success = True
                except Exception as e:
                    retry_count += 1
                    if progress_callback:
                        progress_callback(0, 100, f"Échec batch {i//batch_size + 1}, tentative {retry_count}/{max_retries}")
                    if retry_count < max_retries:
                        time.sleep(retry_delay)  # Attend avant de réessayer

            if not success:
                if progress_callback:
                    progress_callback(0, 100, f"Échec définitif batch {i//batch_size + 1}")
                return False

            if progress_callback:
                current_point = min(i + batch_size, len(points))
                progress_callback(60 + int(40 * current_point / len(points)),
                                100,
                                f"Indexation : {current_point}/{len(points)} chunks")

        if progress_callback:
            progress_callback(100, 100, "Indexation terminée avec succès")
        return True

    except Exception as e:
        if progress_callback:
            progress_callback(0, 100, f"Erreur : {str(e)}")
        st.error(f"Erreur dans process_and_index_document: {e}")
        return False

# --- 2b. Fonctions de recherche ---

# Fonction qui supprime les accents
def normalize_query(query: str) -> str:
    # Supprime les accents et normalise en ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', query)
        if unicodedata.category(c) != 'Mn'
    )

# Fonction qui extrait le sujet principal de la question
def extract_subject(question: str) -> str:
    """
    Extrait le sujet principal d'une question parlementaire sous forme de 3 à 5 mots-clés.
    - Supprime toute mention de députés, du Gouvernement ou de formulations inutiles.
    - Nettoie la ponctuation et tronque à quelques mots.
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",  # possibilité de remplacer "small" par "medium"
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui identifie uniquement le sujet principal "
                    "d'une question parlementaire. "
                    "Ne mentionne jamais le député ou le gouvernement. "
                    "Donne une réponse sous forme de 3 à 5 mots-clés concis, "
                    "centrés sur le thème (pas de phrase complète)."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.2,
        "max_tokens": 20
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    try:
        subject = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        subject = "sujet non identifié"

    # Nettoyage supplémentaire côté Python
    subject = re.sub(r"\b(députée?|député|gouvernement|ministre|assemblée nationale|sénat)\b", "", subject, flags=re.IGNORECASE)
    subject = subject.replace("**Sujet principal**", "").strip()
    subject = re.sub(r"[^\w\s]", " ", subject)  # supprime ponctuation
    subject = re.sub(r"\s+", " ", subject)

    # Tronquer à 5 mots max
    tokens = subject.split()
    subject = " ".join(tokens[:5])

    print("=== Sujet extrait (compact) ===", subject)

    return subject

# Fonction de recherche Tavily (et filtre pour les scores de pertinence <0.5)
def search_tavily_government(subject: str, min_score: float = 0.5):
    """
    Recherche les annonces gouvernementales récentes sur un sujet donné via Tavily.
    - Filtre par domaines autorisés
    - Filtre par date (moins d'un an si disponible)
    - Filtre par score de pertinence (>= min_score)
    """

    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}

    allowed_domains = [
        "gouvernement.fr", "education.gouv.fr", "vie-publique.fr", "elysee.fr",
        "solidarites.gouv.fr", "sante.gouv.fr", "travail-sante-solidarites.gouv.fr",
        "securite-sociale.fr", "ameli.fr", "lassuranceretraite.fr", "caf.fr",
        "msa.fr", "urssaf.fr", "legifrance.gouv.fr", "drees.solidarites-sante.gouv.fr",
        "ars.sante.fr", "cnsa.fr", "en3s.fr", "francetravail.fr"
    ]

    payload = {
        "query": f"dernières annonces gouvernement France {subject}",
        "max_results": 100,
        "include_answer": True,
        "include_domains": allowed_domains
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    raw_results = data.get("results", [])

    # Filtre par domaine
    by_domain = [r for r in raw_results if any(d in r.get("url", "") for d in allowed_domains)]

    # Filtre par date (moins d'un an si dispo)
    cutoff = datetime.now() - timedelta(days=365)
    recent = []
    for r in by_domain:
        ds = r.get("published_date") or r.get("date")
        if ds:
            try:
                pub = datetime.fromisoformat(ds.replace("Z", ""))
                # ✅ Condition supplémentaire : année 2025
                if pub.year == 2025 or pub >= cutoff:
                    recent.append(r)
                    continue
                else:
                    continue
            except Exception:
                # date non exploitable → on garde
                recent.append(r)
        else:
            recent.append(r)

    # Filtre par score
    filtered = [r for r in recent if r.get("score", 0) >= min_score]
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

    # ✅ Limiter à 10 après filtrage
    data["results"] = filtered[:10]

    return data

# Fonction de recherche Google
def search_google_government(subject: str,
                             min_score: float = 0.5,
                             max_results: int = 10):
    """
    Recherche des informations via Google Custom Search API sur un sujet donné.
    - Même structure et format que search_tavily_government
    - Entrée: subject (str), min_score, max_results
    - Sortie: dict {"results": [{"title","url","content","score","published_date"}]}
    """

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": f"dernières annonces gouvernement France {subject}",
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "num": max_results
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    raw_results = data.get("items", [])
    allowed_domains = [
        "gouvernement.fr", "info.gouv.fr", "elysee.fr", "vie-publique.fr",
        "education.gouv.fr", "solidarites.gouv.fr", "sante.gouv.fr",
        "travail-sante-solidarites.gouv.fr", "securite-sociale.fr", "ameli.fr",
        "lassuranceretraite.fr", "caf.fr", "msa.fr", "urssaf.fr",
        "legifrance.gouv.fr", "drees.solidarites-sante.gouv.fr", "ars.sante.fr",
        "cnsa.fr", "en3s.fr", "francetravail.fr"
    ]

    # Filtre par domaine
    by_domain = [r for r in raw_results if any(d in r.get("link", "") for d in allowed_domains)]

    # Filtre par date (moins d'un an si dispo)
    cutoff = datetime.now() - timedelta(days=365)
    recent = []
    for r in by_domain:
        date_str = r.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time")
        if date_str:
            try:
                pub = datetime.fromisoformat(date_str.replace("Z", ""))
                if pub >= cutoff:
                    recent.append(r)
            except Exception:
                recent.append(r)
        else:
            recent.append(r)

    # Filtre par score (Google ne fournit pas de score → fallback = 1.0 si snippet présent)
    filtered = [r for r in recent if r.get("snippet")]
    filtered = filtered[:max_results]

    # Format homogène comme Tavily
    results = []
    for r in filtered:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "content": r.get("snippet", ""),   # Tavily utilisait "content"
            "score": 1.0,                      # Valeur par défaut
            "published_date": date_str if r.get("pagemap", {}).get("metatags") else None
        })

    return {"results": results}

# --- Fonction de log pour le debug ---
def log_debug(title: str, data: Any, max_length: int = 500):
    """Affiche un log de debug dans un fichier."""
    import os
    from datetime import datetime

    # Chemin absolu vers le dossier de logs (dans le répertoire courant)
    log_dir = os.path.join(os.getcwd(), "logs")

    # Créer le dossier s'il n'existe pas
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"📁 Dossier créé : {log_dir}")

    # Chemin absolu vers le fichier de log
    log_file = os.path.join(log_dir, "debug_logs.txt")

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- {title} ---\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    f.write(f"{k}: {str(v)[:max_length]}\n")
            elif isinstance(data, list):
                f.write(f"List of {len(data)} items:\n")
                for i, item in enumerate(data[:3]):
                    f.write(f"  Item {i}: {str(item)[:max_length]}\n")
            else:
                f.write(f"{str(data)[:max_length]}\n")
            f.write("---\n")

        # Confirmation dans le terminal
        print(f"✅ Log enregistré dans {log_file}: {title}")

    except Exception as e:
        print(f"❌ Erreur lors de l'écriture du log : {str(e)}")

# Fonctions utilitaires car le champ base_legislative peut contenir soit des objets BaseLegislativeRef, soit des dicts (selon la provenance des données)
def get_ref_uid(ref) -> str | None:
    return ref.uid if hasattr(ref, "uid") else ref.get("uid")

def get_ref_collection(ref, default_collection: str) -> str:
    return ref.collection if hasattr(ref, "collection") else ref.get("collection", default_collection)

# Préparer pour préparer le contenu en vue d'un export .txt
def build_export_content(response_data: dict, mode: str, include_legal_articles: bool = False) -> str:
    lines = []

    # Question
    question = response_data.get("question")
    if question:
        lines.append("❓ Question\n")
        lines.append(str(question))
        lines.append("\n\n")

    # En-tête (Réponse ou Analyse)
    if mode == "analyse":
        lines.append("🔎 Analyse juridique\n")
    else:
        lines.append("📜 Réponse\n")
    lines.append(str(response_data.get("response", "Pas de texte généré")))
    lines.append("\n\n")

    # Résumé (si présent)
    summary = response_data.get("summary")
    if summary:
        lines.append("📝 Résumé\n")
        lines.append(str(summary))
        lines.append("\n\n")

    # Résultats de recherche (si présents)
    search_results = response_data.get("search_results", [])
    if search_results:
        lines.append("🌐 Résultats de recherche\n")
        for idx, item in enumerate(search_results, start=1):
            titre = item.get("title", "Sans titre")
            url = item.get("url", "")
            extrait = item.get("content") or item.get("snippet") or ""
            score = item.get("score", "N/A")
            date = item.get("published_date", "N/A")
            lines.append(f"{idx}. {titre}\n")
            if url:
                lines.append(f"   Lien: {url}\n")
            if extrait:
                lines.append(f"   Extrait: {extrait}\n")
            lines.append(f"   Score: {score} | Date: {date}\n\n")

    # Anciennes QE (si présentes)
    lines.append("🏛️ Anciennes QE\n")
    for doc in response_data.get("similar_documents", []):
        score = f"{getattr(doc, 'score', 0):.2f}" if getattr(doc, "score", None) is not None else "N/A"
        chambre = getattr(doc, "chambre", "Inconnue")
        lines.append(f"- QE {doc.uid} ({chambre}) - Score : {score}\n")
        lines.append(f"  Question: {doc.question}\n")
        lines.append(f"  Réponse: {doc.reponse}\n\n")

    # Articles juridiques (pour les deux modes)
    if mode == "analyse" or include_legal_articles:
        lines.append("⚖️ Articles juridiques\n")

        # Mode "Analyse juridique" : utilise "sources" (avec relations parents/enfants)
        if mode == "analyse":
            sources = response_data.get("sources", [])
            if not sources:
                lines.append("Aucun article juridique enregistré.\n")
            else:
                for source in sources:
                    art = source["article"]
                    lines.append(f"--- Article {art.num} ({art.collection}) ---\n")
                    lines.append(f"Titre: {art.titre}\n")
                    lines.append(f"Texte: {art.article_complet}\n")

                    # Relations parents (articles cités)
                    if source.get("parents"):
                        lines.append("Articles cités:\n")
                        for parent in source["parents"]:
                            lines.append(f"- {parent.num}: {parent.titre}\n")

                    # Relations enfants (référencé par)
                    if source.get("enfants"):
                        lines.append("Référencé par:\n")
                        for enfant in source["enfants"]:
                            lines.append(f"- {enfant.num}: {enfant.titre}\n")
                    lines.append("\n")

        # Mode "Réponse parlementaire" : utilise "legal_sources" (sans relations)
        else:
            legal_sources = response_data.get("legal_sources", [])
            if not legal_sources:
                lines.append("Aucun texte juridique cité.\n")
            else:
                for art in legal_sources:
                    lines.append(f"- Article {getattr(art, 'num', 'N/A')} ({getattr(art, 'collection', 'N/A')})\n")
                    if getattr(art, "titre", None):
                        lines.append(f"  Titre: {art.titre}\n")
                    contenu = art.article_complet if hasattr(art, 'article_complet') else getattr(art, 'contenu', '')
                    lines.append("  Texte:\n" + str(contenu) + "\n\n")

    return "\n".join(str(x) for x in lines)

# Fonction qui construit un objet RetrievedLegalDocument à partir d'un payload Qdrant
def make_retrieved_document(payload: dict, uid: str, collection: str, score: float = 0) -> RetrievedLegalDocument:
    """Construit un objet RetrievedLegalDocument à partir d'un payload Qdrant."""
    return RetrievedLegalDocument(
        chunk_id=uid,
        num=payload.get("num", ""),
        titre=payload.get("titre", ""),
        contenu=payload.get("contenu", ""),
        article_complet=payload.get("article_complet", payload.get("contenu", "")),
        contexte_hierarchique=payload.get("contexte_hierarchique", ""),
        collection=collection,
        partie=payload.get("partie"),
        livre=payload.get("livre"),
        titre_structure=payload.get("titre_structure"),
        chapitre=payload.get("chapitre"),
        section=payload.get("section"),
        sous_section=payload.get("sous_section"),
        paragraphe=payload.get("paragraphe"),
        sous_paragraphe=payload.get("sous_paragraphe"),
        base_legislative=payload.get("base_legislative", []),
        score=score
    )

# Fonction qui recherche des articles juridiques dans les collections Qdrant en utilisant des embeddings
def search_articles(
    query: str,
    partie: Optional[str] = None,
    limit: int = 5,
    must_contain: Optional[str] = None,
    debug: bool = False,
    threshold: float = 0.0
) -> Dict[str, Any]:
    """Recherche optimisée d'articles juridiques dans Qdrant."""
    target_collections = ["CASF", "Code du travail", "Code de la santé publique", "Code de la sécurité sociale"]
    collections = qdrant_client.get_collections()
    valid_collections = [c.name for c in collections.collections if c.name in target_collections]

    if not valid_collections:
        return {"sources": [], "total": 0, "limit": limit, "offset": 0}

    try:
        # Recherche vectorielle
        embedding = embedding_model.encode(query).tolist()
        query_filter = models.Filter(
            must=[models.FieldCondition(key="partie", match=models.MatchValue(value=partie))]
        ) if partie else None

        all_results = []
        for collection in valid_collections:
            try:
                hits = qdrant_client.search(
                    collection_name=collection,
                    query_vector=embedding,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                all_results.extend(hits)
            except Exception as e:
                continue

        # Filtrage par score
        results = [r for r in all_results if r.score >= threshold]

        # Filtrage par mot-clé
        if must_contain:
            results = [
                r for r in results
                if must_contain.lower() in r.payload.get("contenu", "").lower()
            ]

        if not results:
            return {"sources": [], "total": 0, "limit": limit, "offset": 0}

        # Mapping vers RetrievedLegalDocument
        documents: List[RetrievedLegalDocument] = []
        for result in results:
            try:
                # Utilise payload["chunk_id"] (str) au lieu de result.id (int/UUID)
                chunk_id = result.payload.get("chunk_id")  # Ex: "L111-1_chunk0"
                if not chunk_id:
                    chunk_id = result.payload.get("uid", str(result.id))  # Fallback pour les collections sans chunk_id
                article = make_retrieved_document(
                    payload=result.payload,
                    uid=chunk_id,  # ← Ici, chunk_id est TOUJOURS une chaîne
                    collection=result.payload.get("collection", ""),
                    score=result.score
                )
                documents.append(article)
            except Exception as e:
                continue

        # Déduplication par numéro
        seen = set()
        unique_documents = []
        for doc in documents:
            if doc.num not in seen:
                seen.add(doc.num)
                unique_documents.append(doc)

        # Tri par numéro
        unique_documents.sort(key=lambda d: d.num)

        return {
            "sources": unique_documents[:limit],
            "total": len(unique_documents),
            "limit": limit,
            "offset": 0
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"sources": [], "total": 0, "limit": limit, "offset": 0}

# Fonction pour construire un arbre législatif
def build_legislative_tree(sources: List[RetrievedLegalDocument]) -> Dict[str, Any]:
    """
    Version corrigée qui:
    1. Évite de modifier le dictionnaire pendant l'itération
    2. Utilise une copie des clés pour l'itération
    3. Conserve tous les logs et la structure uniforme
    """

    # 1. Ajouter tous les articles sources (déjà des objets)
    enrichis = {}
    for art in sources:
        enrichis[art.num] = {
            "article": art,
            "parents": [],
            "enfants": [],
            "type": art.num[0] if art.num else "?"
        }

    # 2. Pour chaque article source, récupérer les articles de même groupe hiérarchique
    for art in sources:
        # Priorité des niveaux hiérarchiques (du plus précis au plus général)
        hierarchy_levels = [
            ("sous_paragraphe", art.sous_paragraphe),
            ("paragraphe", art.paragraphe),
            ("sous_section", art.sous_section),
            ("section", art.section),
            ("chapitre", art.chapitre)
        ]

        for level_key, level_value in hierarchy_levels:
            if not level_value:
                continue

            try:
                flt = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=level_key,
                            match=models.MatchText(text=level_value)
                        ),
                        models.FieldCondition(
                            key="collection",
                            match=models.MatchValue(value=art.collection)
                        )
                    ]
                )

                points, _ = qdrant_client.scroll(
                    collection_name=art.collection,
                    scroll_filter=flt,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload
                    art_num = payload.get("num")
                    if not art_num or art_num in enrichis:
                        continue

                    new_art = make_retrieved_document(
                        payload=payload,
                        uid=payload.get("chunk_id", str(point.id)),
                        collection=payload.get("collection", art.collection),
                        score=0
                    )
                    enrichis[art_num] = {
                        "article": new_art,
                        "parents": [],
                        "enfants": [],
                        "type": art_num[0] if art_num else "?"
                    }

                break

            except Exception as e:
                continue

    # 3. Gestion des références (base_legislative) pour TOUS les articles
    # ✅ Solution: Itérer sur une COPIE des clés pour éviter de modifier le dict pendant l'itération
    for art_uid in list(enrichis.keys()):  # ← COPIE des clés !
        art_data = enrichis[art_uid]
        art = art_data["article"]

        for ref in art.base_legislative or []:
            ref_uid = ref.uid if hasattr(ref, 'uid') else ref.get('uid')
            if not ref_uid or ref_uid in enrichis:
                continue

            try:
                ref_collection = ref.collection if hasattr(ref, 'collection') else art.collection

                ref_points, _ = qdrant_client.scroll(
                    collection_name=ref_collection,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="num",
                            match=models.MatchValue(value=ref_uid)
                        )]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )

                if ref_points:
                    ref_payload = ref_points[0].payload
                    ref_art = make_retrieved_document(
                        payload=ref_payload,
                        uid=ref_payload.get("chunk_id", str(ref_points[0].id)),
                        collection=ref_payload.get("collection", ref_collection),
                        score=0
                    )
                    enrichis[ref_uid] = {  # ✅ Ajout d'un nouvel élément au dict
                        "article": ref_art,
                        "parents": [],
                        "enfants": [],
                        "type": ref_uid[0] if ref_uid else "?"
                    }

                    # Ajout des relations parent/enfant
                    art_data["parents"].append(enrichis[ref_uid]["article"])
                    enrichis[ref_uid]["enfants"].append(art_data["article"])

            except Exception as e:
                continue

    return enrichis

# Fonction de tri des numéros
def sort_key_num(article_data: Dict[str, Any]) -> tuple:
    """Clé de tri pour les numéros d'article (ex: L241-3)."""
    num = article_data["article"].num  # Accès direct à l'attribut Pydantic
    m = re.match(r"[LRD](\d+)(?:-(\d+))?", num)
    if m:
        base = int(m.group(1))
        suffix = int(m.group(2)) if m.group(2) else 0
        return (base, suffix)
    return (float("inf"), float("inf"))

# Fonction qui ajoute un article dans l'arbre partagé selon les niveaux hiérarchiques
def add_to_tree(tree: dict, article: RetrievedLegalDocument, item: dict):
    """
    Ajoute un article dans l'arbre hiérarchique avec ses relations.
    - article: Objet RetrievedLegalDocument (pour la hiérarchie)
    - item: Dictionnaire COMPLET avec parents/enfants (pour les relations)
    """
    # Construire la hiérarchie depuis les champs de l'article
    levels = [
        article.collection,
        article.partie,
        article.livre,
        article.titre_structure,
        article.chapitre,
        article.section,
        article.sous_section,
        article.paragraphe,
        article.sous_paragraphe,
    ]
    # Filtrer les niveaux vides
    levels = [lvl for lvl in levels if lvl]

    # Naviguer dans l'arbre
    node = tree
    for label in levels:
        node = node.setdefault(label, {})

    # Ajouter l'article AVEC SES RELATIONS
    node.setdefault("_items", []).append(item)  # ✅ item contient parents/enfants

# Fonction qui affiche récursivement l'arbre sous forme d'expanders
def render_tree(container, node: dict, level: int = 0):
    """
    Version simplifiée qui affiche :
    1. Tous les articles dans une arborescence unique
    2. Pour chaque article : uniquement "Référencé par" et "Articles cités"
    """
    # Styles CSS
    container.markdown("""
    <style>
        .article-title { font-weight: bold; margin-bottom: 5px; }
        .article-body { font-size: 14px; line-height: 1.4; margin-bottom: 10px; }
        .relation-section { margin-top: 10px; margin-bottom: 10px; }
        .relation-title { font-weight: bold; color: #333; }
        .relation-item { margin-left: 15px; color: #555; }
    </style>
    """, unsafe_allow_html=True)

    # Affichage des articles avec tri numérique
    for data in sorted(node.get("_items", []),
                      key=lambda x: [
                          int(part) if part.isdigit() else part
                          for part in x["article"].num.split('-')
                      ]):
        art = data["article"]

        with container.expander(f"📜 {art.num} - {art.titre}"):
            # Contenu de l'article
            container.markdown(f'<div class="article-body">{art.article_complet}</div>', unsafe_allow_html=True)

            # Section "Référencé par" (anciennement "Enfants")
            if data.get("enfants"):
                container.markdown('<div class="relation-section">'
                                   '<div class="relation-title">👶 Référencé par :</div>', unsafe_allow_html=True)
                for enfant in data["enfants"]:
                    container.markdown(f'<div class="relation-item">- {enfant.num} : {enfant.titre}</div>',
                                      unsafe_allow_html=True)

            # Section "Articles cités" (anciennement "Parents")
            if data.get("parents"):
                container.markdown('<div class="relation-section">'
                                   '<div class="relation-title">📚 Articles cités :</div>', unsafe_allow_html=True)
                for parent in data["parents"]:
                    container.markdown(f'<div class="relation-item">- {parent.num} : {parent.titre}</div>',
                                      unsafe_allow_html=True)

    # Navigation hiérarchique (uniquement pour l'organisation visuelle)
    for label, child in sorted(((k, v) for k, v in node.items() if k != "_items"), key=lambda x: x[0]):
        with container.expander(f"📁 {label}"):
            render_tree(container, child, level + 1)

# Fonction de recherches de documents dans tout le RAG (hors codes et jeu de données QE) pour alimenter l'API
def search_uploaded_documents(
    query: str,
    qdrant_client: Any,
    embedding_model: Any,
    selected_collections: List[str] = None,
    top_k: int = 3,
    top_k_selected = 10,
) -> List[Dict]:
    """
    Recherche dans les collections-documents (1 collection = 1 document).
    Args:
        query: Requête de recherche.
        qdrant_client: Client Qdrant.
        embedding_model: Modèle d'embedding.
        selected_collections: Liste des collections à rechercher (optionnel).
        top_k: Nombre de résultats max si aucune collection sélectionnée.
        top_k_selected: Nombre de résultats max si collections sélectionnées.
    Returns:
        Liste de dicts avec les champs essentiels.
    """
    query_embedding = embedding_model.encode(query).tolist()
    all_results = []
    try:
        # 1. Récupère les collections à rechercher
        protected = {
            "QuestionParlementaire",
            "Code de la sécurité sociale",
            "Code du travail",
            "CASF",
            "Code de la santé publique",
        }
        collections = qdrant_client.get_collections()
        doc_collections = [col.name for col in collections.collections if col.name not in protected]

        # 2. Limite aux collections sélectionnées si spécifiées
        if selected_collections:
            doc_collections = [col for col in doc_collections if col in selected_collections]
            # le nombre de chunks retournés passe à 10 si la recherche est limité sur une ou plusieurs collections
            top_k = top_k_selected
            if not doc_collections:
                doc_collections = [col for col in doc_collections if col not in protected]
            
        # 3. Recherche dans chaque collection-document
        for collection in doc_collections:
            try:
                results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                for result in results:
                    payload = result.payload or {}
                    all_results.append({
                        "collection": collection,
                        "text": payload.get("text", ""),
                        "score": float(result.score) if hasattr(result, "score") else None,
                        "title": payload.get("section"),
                    })
            except Exception as e:
                st.warning(f"Erreur sur {collection}: {e}")

        # 4. Trie par score et limite les résultats
        all_results.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
        return all_results[:top_k]

    except Exception as e:
        st.error(f"Erreur de recherche dans les documents uploadés: {e}")
        return []

# Fonction qui ajuste la taille du contexte issu des "autres collections" à la pertinence (score) du retrieval
def format_uploaded_docs_by_relevance(
    uploaded_results: List[Dict],
    min_score: float = 0.7,
    max_docs: int = 3,
    max_length: int = 400
) -> str:
    """
    Trie les résultats par score et formate les extraits textuels pour enrichir le prompt.
    - uploaded_results : liste de dictionnaires renvoyés par la recherche vectorielle
    - min_score : seuil de pertinence minimum
    - max_docs : nombre maximum de documents à inclure
    - max_length : longueur maximale de l'extrait
    """
    if not uploaded_results:
        return ""

    # Filtrer par score
    filtered = [doc for doc in uploaded_results if doc.get("score", 0) >= min_score]

    # Trier par score décroissant
    filtered.sort(key=lambda d: d.get("score", 0), reverse=True)

    # Limiter le nombre de docs
    filtered = filtered[:max_docs]

    formatted = []
    for doc in filtered:
        passage = doc.get("text", "")
        source = doc.get("collection", "inconnu").replace("_", " ")
        title = doc.get("title", "") or ""
        score = doc.get("score", 0)

        formatted.append(
            f"Source: {source} (Section: {title})\n"
            f"Passage: {passage[:max_length]}...\n"
            f"(Score: {score:.2f})\n"
        )

    return "\n---\n".join(formatted)

# Fonction de recherches d'anciennes questions / réponses dans le RAG Qdrant
def search_question_parlementaire(query: str, top_k: int = 5) -> List[ResponseDocument]:
    embedding = embedding_model.encode(query).tolist()
    hits = qdrant_client.search(
        collection_name="QuestionParlementaire",
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    results: List[ResponseDocument] = []

    for i, r in enumerate(hits):
        p = r.payload or {}
        try:
            # Normalisation sécurisée des champs
            uid = str(p.get("uid", "")) if p.get("uid") is not None else ""
            legislature = str(p.get("legislature")) if p.get("legislature") is not None else None
            ministeres = p.get("ministeres_attribues")
            if isinstance(ministeres, str):
                ministeres = [ministeres]
            elif not isinstance(ministeres, list):
                ministeres = []
            textes_juridiques = p.get("textes_juridiques")
            if isinstance(textes_juridiques, str):
                textes_juridiques = [textes_juridiques]
            elif not isinstance(textes_juridiques, list):
                textes_juridiques = []

            # Création du document
            doc = ResponseDocument(
                uid=uid,
                question=p.get("question", ""),
                reponse=p.get("reponse", ""),
                legislature=legislature,
                chambre=p.get("chambre"),
                rubrique=p.get("rubrique"),
                analyse=p.get("analyse"),
                ministeres_attribues=ministeres,
                date_question=p.get("date_question"),
                date_reponse=p.get("date_reponse"),
                textes_juridiques=textes_juridiques,
                score=r.score
            )
            results.append(doc)

        except Exception as e:
             continue  # Passe au résultat suivant

    return results

# Fonction qui extrait un ordre numérique à partir d'un label en chiffres romains
ROMAN_MAP = {
    "I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10,
    "XI":11,"XII":12,"XIII":13,"XIV":14,"XV":15,"XVI":16,"XVII":17,"XVIII":18,"XIX":19,"XX":20
}
def extract_order(label: str) -> int:
    if not label:
        return float("inf")
    m_roman = re.search(r"\b(Ier|[IVXLCDM]+)\b", label)
    if m_roman:
        return 1 if m_roman.group(1) == "Ier" else ROMAN_MAP.get(m_roman.group(1), float("inf"))
    m_num = re.search(r"\b(\d+)\b", label)
    if m_num:
        return int(m_num.group(1))
    return float("inf")

# Construit un prompt pour Mistral Large afin de générer une analyse juridique.
def build_legal_analysis_prompt(question: str, articles: List[RetrievedLegalDocument], stats: dict) -> str:
    # Construit un prompt pour Mistral Large afin de générer une analyse juridique.
    # Préparation des articles sous forme de contexte avec troncature brute
    max_chars = 70000  # limite totale pour les articles (≈ 3500-4000 tokens)
    articles_context = []
    total_chars = 0

    for article in articles:
        uid = article.num
        titre = article.titre
        contenu = article.article_complet  # ou article.contenu selon votre besoin

        if total_chars + len(contenu) > max_chars:
            # Tronquer le dernier article pour ne pas dépasser la limite
            allowed = max_chars - total_chars
            truncated = contenu[:allowed]
            articles_context.append(f"### Article {uid}: {titre}\n{truncated}\n[Texte tronqué]\n")
            break
        else:
            articles_context.append(f"### Article {uid}: {titre}\n{contenu}\n")
            total_chars += len(contenu)

    articles_str = "\n".join(articles_context)

    # Consignes pour Mistral
    prompt = f"""
    [INST]
    **Consignes strictes pour une analyse juridique complète :**
    1. **Structure obligatoire** à respecter impérativement :
       - Introduction (50-100 mots) : rappel du contexte juridique de la question
       - Analyse détaillée (80-90% du contenu) :
         * L'analyse doit être faite exclusivement à partir des articles suivants : {articles_str}
         * Présentation des principes généraux avec citations précises d'un maximum d'articles
         * Présentation des enjeux
       - Conclusion synthétique (100-150 mots)
       - Toute réponse qui se termine par une phrase tronquée sera considérée comme irrecevable

    2. **Exigences de complétude** :
       - Toute phrase doit être grammaticalement complète
       - La dernière phrase doit impérativement résumer un point clé
       - Si le développement dépasse la limite, prioriser :
         1. Les principes fondamentaux
         2. Les exceptions majeures
         3. Un exemple concret d'application

    3. **Style requis** :
       - Style très concis
       - Phrases courtes (20-25 mots max)
       - Un paragraphe = une idée juridique précise
       - Citations systématiques des articles (ex: "L'article R241-12 précise que...")
       - Éviter les formules vagues ("certains cas", "parfois") → préciser les conditions

    **Question à analyser** :
    {question}

    [/INST]
    """
    return prompt

# Appelle l'API Mistral Large pour générer une analyse juridique ##### 4000 TOKENS DEFINIS ICI ######
def call_mistral_legal_analysis(
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    model_size: str = "small"  # "large", "medium", "small"
):
    # Appelle l'API Mistral avec chainage automatique pour les réponses tronquées.
    mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construire le nom du modèle dynamiquement
    model_name = f"mistral-{model_size}-latest"

    def is_complete(response_text):
        # Vérifie si une réponse est grammaticalement complète
        if not response_text:
            return False

        last_char = response_text[-1]
        last_sentence = response_text.split('.')[-1].strip()

        return (
            (last_char in ('.', '!', '?')) and
            (len(last_sentence.split()) > 3) and
            (not last_sentence.endswith((':', ';', ','))) and
            (not response_text.endswith(('...', '–', '—')))
        )

    def complete_response(truncated_response):
        # Termine une réponse tronquée
        completion_prompt = f"""
        [INST]
        Terminez cette analyse juridique de manière complète et professionnelle.
        Analyse en cours: "{truncated_response[-500:]}"  # Derniers 500 caractères

        Consignes strictes:
        1. Résumez en 1 phrase le point juridique en cours
        2. Ajoutez 2-3 phrases de conclusion qui:
           - Synthétisent les points clés
           - Proposent une application pratique
           - Se terminent impérativement par un point
        3. Utilisez un style formel: "En conséquence...", "Ainsi, il ressort que...", etc.
        [/INST]
        """

        try:
            completion_response = requests.post(
                mistral_api_url,
                headers=headers,
                json={
                    "model": model_name,  # <-- correction ici
                    "messages": [{"role": "user", "content": completion_prompt}],
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "stop": ["."]  
                },
                timeout=30
            )
            completion_response.raise_for_status()
            completion = completion_response.json()["choices"][0]["message"]["content"].strip()

 #            if is_complete(completion):
            return truncated_response + "\n\n" + completion
 #            else:
 #                return truncated_response + "\n\nCette analyse couvre succinctement les principaux aspects juridiques de la question posée."

        except Exception as e:
            return truncated_response + f"\n\n[Note: La conclusion de cette analyse a été synthétisée. Erreur technique: {str(e)}]"

    try:
        response = requests.post(
            mistral_api_url,
            headers=headers,
            json={
                "model": model_name,  # <-- correction ici aussi
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        analysis = data["choices"][0]["message"]["content"].strip()

        if not is_complete(analysis):
            analysis = complete_response(analysis)

        return analysis

    except requests.exceptions.RequestException as e:
        return f"Erreur lors de l'appel à l'API Mistral: {str(e)}"

# Fonction de génération d'analyse juridique
def generate_legal_analysis(
    question: str,
    must_contain: str = "",
    max_articles: int = 5,
    threshold: float = 0.5,
    model_size: str = "small",
    search_button: bool = False,
    generate_analysis_button: bool = False
) -> dict:
    try:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Recherche des articles (avec cache)
        if "last_articles_search" not in st.session_state or search_button:
            st.session_state.last_articles_search = search_articles(
                query=question,
                limit=max_articles,
                must_contain=must_contain if must_contain else None,
                debug=True,
                threshold=threshold
            )
        articles = st.session_state.last_articles_search

        # Préparation des sources enrichies
        enrichis = build_legislative_tree(articles["sources"])
        stats = enrichis.pop('stats', {})

        # Formatage des sources pour l'affichage et l'analyse
        sources = []
        for uid, data in enrichis.items():
            if isinstance(data, dict) and "article" in data:
                sources.append({
                    "uid": uid,
                    "article": data["article"],
                    "type": data.get("type"),
                    "parents": data.get("parents", []),
                    "enfants": data.get("enfants", [])
                })

        # Fallback si aucun enrichissement
        if not sources:
            sources = [{
                "uid": art.num,
                "article": art,
                "type": art.num[0] if art.num else "?",
                "parents": [],
                "enfants": []
            } for art in articles["sources"]]

        # Gestion des cas d'erreur
        if not sources or len(sources) == 0:
            st.warning("Aucun article juridique valide après traitement.")
            return {
                "response": "Aucun article juridique valide après traitement.",
                "sources": [],
                "similar_documents": [],
                "debug_logs": captured_output.getvalue(),
                "stats": stats
            }

        # Mode "Recherche" (affichage des articles)
        if search_button:
            return {
                "response": "Voir les articles dans l'onglet sources.",
                "sources": sources,
                "similar_documents": [],
                "debug_logs": captured_output.getvalue(),
                "stats": stats
            }

        # Mode "Génération" (appel à Mistral)
        elif generate_analysis_button:
            prep_placeholder = st.empty()
            prep_placeholder.markdown(
                '''
                <div class="prep-message">
                    🔧 Production de l'analyse en cours<span id="dots">...</span>
                </div>
                <script>
                    const dots = document.getElementById('dots');
                    let dotCount = 0;
                    const interval = setInterval(() => {
                        dotCount = (dotCount + 1) % 4;
                        dots.textContent = '.'.repeat(dotCount);
                    }, 500);
                </script>
                ''',
                unsafe_allow_html=True
            )

            # Appel à Mistral avec les articles originaux (non enrichis)
            prompt = build_legal_analysis_prompt(question, articles["sources"], stats)
            legal_analysis = call_mistral_legal_analysis(
                prompt,
                max_tokens=4000,
                temperature=0.3,
                model_size=model_size
            )

            # Efface le message de chargement
            prep_placeholder.empty()

            return {
                "response": legal_analysis,
                "sources": sources,  # Sources enrichies pour l'affichage
                "similar_documents": [],
                "debug_logs": captured_output.getvalue(),
                "stats": stats
            }

    finally:
        sys.stdout = old_stdout

# Construit le prompt d'appel à Mistral
def build_parlementary_response_prompt(
    question: str,
    parliamentary_context: str,
    legal_context: str,
    uploaded_documents: str,
    detail_juridique: int,
    longueur: str,
    response_orientation: str,
    custom_instructions: str,
    search_context: str
) -> str:
    # Construit un prompt optimisé avec contexte parlementaire ET juridique.
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
            "Exemple : 'Plutôt qu’une refonte complète du dispositif, nous avons renforcé [mesure X], qui a déjà permis [résultat].' "
            "Évitez les formulations défensives comme 'nous ne pouvons pas' – préférez 'notre approche privilégie [solution], car [raison].'",

        "Répondre positivement aux propositions du parlementaire":
            "Saluiez l’intérêt de la proposition **sans reprendre les critiques sous-jacentes**. "
            "Utilisez des formulations comme : "
            "'Votre proposition s’inscrit dans une dynamique que le Gouvernement partage, comme en attestent [mesures existantes].' "
            "'Nous partageons votre préoccupation pour [enjeu], et nos actions vont dans le sens de [objectif], comme le montre [exemple].' "
            "Évitez : 'Vous avez raison de souligner que...' → préférez : 'Votre attention à ce sujet rejoint nos priorités, illustrées par [action].'",

        "Répondre de manière technique et détaillée":
            "Fournissez une réponse **factuelle et technique**, en évitant tout commentaire sur les critiques du parlementaire. "
            "Structurez ainsi : "
            "1. **Cadre juridique** : 'Le dispositif actuel, défini par [article X], repose sur [principe].' "
            "2. **Données chiffrées** : 'Les derniers chiffres (source : [DREES/INSEE/...], [année]) montrent que [tendance].' "
            "3. **Mesures en cours** : 'Pour répondre à ces enjeux, [mesure A] et [mesure B] ont été mises en place, avec [résultat].' "
            "Utilisez un vocabulaire neutre et des verbes d’action : 'le Gouvernement a engagé', 'les services travaillent à', 'les résultats montrent que'."
    }

    # Longueur maximale selon le paramètre
    max_tokens = 500 if longueur.startswith("Courte") else 1000 if longueur.startswith("Moyenne") else 2200

    # Construction du prompt
    prompt = f"""
    [INST]
    {orientation_mapping.get(response_orientation, "")}

    **Question parlementaire :**
    {question}

    **Contexte parlementaire (réponses similaires passées) :**
    {parliamentary_context}

    **Textes juridiques applicables (PRIORITAIRES) :**
    {legal_context}

    **Documents de référence uploadés (traités avec vos fonctions d'extraction) :**
    {uploaded_documents}

    **Résultats de recherche internet (actualités, positions du gouvernement) :**
    {search_context}

    **Consignes strictes :**
    1. **Priorité juridique** : Votre réponse DOIT être cohérente avec les textes juridiques fournis.
       En cas de contradiction entre le contexte parlementaire et les textes juridiques, priorisez ces derniers.
       En cas de contradiction entre le contexte parlementaire et les résultats de recherche internet, priorisez ces derniers.
       Citez explicitement les articles pertinents (ex: "comme le précise l'article L124-5 du CASF...").

    2. **Structure** :
       - Reconnaître l'importance du sujet sans insister sur les difficultés soulevées par le parlementaire, surtout si elles sont critiques quant à l'action du Gouvernement
       - Rappelez éventuellement les chiffres et le cadre juridique
       - Poursuivez avec les éléments budgetaires
       - Intégrez les informations issues prioritairement des documents de référence uploadés puis de la recherche internet de manière fluide, sans mention explicite de la source ("recherche internet", "résultats de recherche") pour :
            - décrire les mesures prises par le Gouvernement et celles sur lesquelles le Gouvernement travaille
            - préciser la position du Gouvernement sur le sujet principal de la question parlementaire
       - Ne pas annoncer d'échéances à venir pour des dates antérieures à la date du jour (exemple : "Une concertation sera menée d’ici l’été 2024" alors que nous sommes en novembre 2025)
       - Concluez en réaffirmant l'engagement du Gouvernement.
       - Ne pas mélanger le sujet à d'autres sujets trop éloignés dans la conclusion.

    3. **Niveau de détail** : {detail_juridique}/5 (adaptez la profondeur des explications juridiques).

    4. **Longueur et ajustement dynamique** :
       - Limite absolue : {max_tokens} tokens.
       - Avant de finaliser, estimez le nombre de tokens de votre réponse.
       - Si vous dépassez {max_tokens} :
            - Supprimez les exemples, les répétitions ou les données secondaires.
            - Conservez impérativement : l’enjeu, le cadre juridique, et la conclusion.
            - Utilisez des formulations comme : "Pour respecter la limite, nous synthétisons les points clés :"
       - Si la réponse risque d’être trop courte, développez le cadre juridique ou les mesures en cours.

    5. **Estimation préalable** :
       - Un paragraphe = ~100 tokens. Adaptez le nombre de paragraphes en conséquence.
       - Après chaque section, vérifiez que le total reste inférieur à {max_tokens}.

    6. **Style** :
       - Utilisez un style administratif, formel et concis, comme dans les réponses ministérielles.
       - La réponse doit être rédigée en prose continue, sans titres, sans puces, sans numérotation.
       - Répondez précisément aux questions posées, par exemple sur les éléments budgétaires ou de calendrier.
       - La réponse doit être d'actualité et privilégier les informations les plus récentes.
       - Si les propositions faites par le parlementaire sont intéressantes, dites qu'elles seront étudiées.
       - Utilisez uniquement des paragraphes rédigés, comme dans les réponses ministérielles publiées au Journal Officiel.
       - Si vous avez plusieurs éléments à présenter, intégrez-les dans des phrases complètes reliées par des connecteurs ("par ailleurs", "en outre", "de plus").
       - Les éléments de la réponse ne doivent pas être redondants.
       - Ne pas mettre de formule de politesse à la fin.
       - **Contrainte de longueur absolue** : La réponse ne doit pas dépasser {longueur}.
       - Toute réponse plus longue sera rejetée.
       - Si le sujet est trop complexe pour tenir dans cette limite, concentrez-vous sur les points les plus importants.
       - Toute réponse qui se termine par une phrase tronquée est incorrecte.
       - Toute réponse qui contient des listes ou des titres est incorrecte.

    {f"7. Instructions spécifiques strictes : {custom_instructions}" if custom_instructions else ""}
    [/INST]
    """
    return prompt

# Appelle l'API Mistral Large pour générer une réponse parlementaire
def call_mistral_parlementary_response(
    prompt: str,
    longueur: str,
    question: str,
    max_retries: int = 2,
    model_size: str = "small"   # "large", "medium", "small"
) -> str:
    # Appelle l'API Mistral pour générer une réponse parlementaire. Le modèle est choisi dynamiquement (small, medium, large).
    mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construire le nom du modèle dynamiquement
    model_name = f"mistral-{model_size}-latest"

    # Déterminer la taille de sortie en fonction de la longueur souhaitée
    if longueur.startswith("Courte"):
        max_tokens = 500
    elif longueur.startswith("Moyenne"):
        max_tokens = 1000
    else:
        max_tokens = 2200

    # Vérifier la taille du prompt
    prompt_tokens = estimate_tokens(prompt)
    if prompt_tokens > 30000:
        raise ValueError(f"Prompt trop long: {prompt_tokens} tokens (limite: 30000)")

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(mistral_api_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()

            mistral_response = data["choices"][0]["message"]["content"]
            is_truncated = data["choices"][0].get("finish_reason") == "length"

            if is_truncated:
                try:
                    mistral_response = handle_truncated_response(mistral_response, question, longueur)
                except Exception as e:
                    raise Exception(f"Erreur lors de la complétion de la réponse tronquée: {str(e)}")

            return mistral_response

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:  # Too Many Requests
                st.warning(f"⏳ API Mistral temporairement encombrée. Tentative {attempt + 1}/{max_retries}. Relance dans 10 secondes...")
                time.sleep(10)
            else:
                raise Exception(f"Erreur HTTP {response.status_code}: {str(e)}")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Erreur réseau. Tentative {attempt + 1}/{max_retries}. Relance dans 10 secondes...")
                time.sleep(10)
            else:
                raise Exception(f"Erreur réseau: {str(e)}")

# Complète une réponse tronquée
def handle_truncated_response(response: str, question: str, longueur: str) -> str:
    # Gère les réponses tronquées par Mistral.
    last_period = response.rfind('.')
    if last_period > 0 and last_period < len(response) - 100:
        incomplete_part = response[last_period+1:]
    else:
        incomplete_part = response[-100:]

    completion_prompt = f"""
    [INST]
    Complétez UNIQUEMENT la phrase ou le paragraphe suivant en cours, sans ajouter de titre ni d'introduction.
    Contexte original: {question[:200]}...
    Texte à compléter: "{incomplete_part}"
    Consignes strictes:
    - Continuez directement le texte existant, sans recommencer la réponse.
    - N'ajoutez pas de formule comme "Réponse du ministère...".- Terminez la phrase/paragraphe en cours de manière cohérente.
    - Ajoutez une conclusion sur le thème de la question en 1-2 phrases maximum.
    - Utilisez un style administratif et formel, comme dans les réponses ministérielles.
    - Respectez strictement la limite de 400 caractères.
    - Tout complément de réponse plus long sera rejeté.
    - Concluez en réaffirmant l'engagement du Gouvernement.
    - La réponse doit être rédigée en prose continue, sans titres, sans puces, sans numérotation.
    - Utilisez uniquement des paragraphes rédigés, comme dans les réponses ministérielles publiées au Journal Officiel.
    - Si vous avez plusieurs éléments à présenter, intégrez-les dans des phrases complètes reliées par des connecteurs ("par ailleurs", "en outre", "de plus").
    - Ne pas mettre de formule de politesse à la fin.
    - Tout complément de réponse qui se termine par une phrase tronquée est incorrecte.
    - Toute complément de réponse qui contient des listes ou des titres est incorrecte.
    [/INST]
    """

    try:
        completion_response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": completion_prompt}],
                "temperature": 0.1,
                "max_tokens": 60
            },
            timeout=30
        )
        completion_response.raise_for_status()
        completion = completion_response.json()["choices"][0]["message"]["content"]

        if response.endswith("..."):
            final_response = response[:-3] + completion
        elif response.endswith(" "):
            final_response = response + completion
        else:
            final_response = response + " " + completion

        if not final_response.endswith(('.', '!', '?')):
            final_response += "."
        return final_response

    except Exception as e:
        st.warning(f"⚠️ Impossible de compléter la réponse tronquée: {str(e)}")
        return response + " (réponse incomplète)"

# --- 7. Génération de la réponse ---
def generate_response(
    question: str,
    legislature: Optional[str] = None,
    rubrique: Optional[str] = None,
    detail_juridique: int = 3,
    longueur: str = "Moyenne (500 mots)",
    response_orientation: str = "Répondre de façon neutre",
    custom_instructions: str = "",
    include_legal_articles: bool = False,
    must_contain: str = "",
    max_legal_articles: int = 3,
    model_size: str = "small"   # choix du modèle Mistral
):
    try:
        status_placeholder = st.empty()

        # Étape 1 : Recherche d'anciennes questions / réponses parlementaires (uniquement si pas de limitation)
        parliamentary_context = "Aucun contexte parlementaire trouvé."
        similar_documents = []
        if not (use_priority_docs and selected_docs):
            status_placeholder.markdown(
                '<div class="status-message">🏛️ Recherche dans la base des anciennes questions / réponses...</div>',
                unsafe_allow_html=True
            )
            similar_documents = search_question_parlementaire(question, top_k=5)
            if similar_documents:
                parliamentary_context = "\n\n".join(
                    [
                        f"Contexte parlementaire {i+1} (source: {doc.uid}):\n"
                        f"Question: {doc.question}\nRéponse: {doc.reponse}"
                        for i, doc in enumerate(similar_documents)
                    ]
                )

        # Étape 2 : Recherche des articles juridiques (si activée) (uniquement si pas de limitation)

        legal_context = "Aucun texte juridique spécifique n'a été identifié."
        legal_sources = []
        if not (use_priority_docs and selected_docs) and include_legal_articles:
            status_placeholder.markdown(
                '<div class="status-message">📚 Recherche dans les codes juridiques (Code du travail, Code de la sécurité sociale, Code de la santé publique, Code de l\'action sociale et des familles)...</div>',
                unsafe_allow_html=True
            )
            legal_sources_result = search_articles(
                query=question,
                partie=None,
                limit=max_legal_articles,
                must_contain=must_contain if must_contain else None,
                debug=True,
                threshold=0.5
            )
            legal_sources = legal_sources_result["sources"]
            if legal_sources:
                legal_context = "\n\n".join(
                    [f"Article {art.num}: {art.titre}\n{art.article_complet}" for art in legal_sources]
                )

        # Étape 3 : Recherche dans les documents uploadés (si mode parlementaire) - limité à 3 résultats
        status_placeholder.markdown(
            '<div class="status-message">🏛️ Recherche dans la base documentaire...</div>',
            unsafe_allow_html=True
        )
        uploaded_results = search_uploaded_documents(question, qdrant_client, embedding_model, top_k=3)
        # Formatage basé sur la pertinence (seuil = 0.7)
        uploaded_docs_context = format_uploaded_docs_by_relevance(uploaded_results, min_score=0.7)

        # Étape 4 : Recherche internet (si mode parlementaire)
        search_context = "Aucune recherche internet effectuée."
        search_results = []

        if not (use_priority_docs and selected_docs) and st.session_state.get("search_engine"):
            if st.session_state.get("search_engine") == "Tavily":
                status_placeholder.markdown(
                    '<div class="status-message">🌐 Recherche internet (Tavily)...</div>',
                    unsafe_allow_html=True
                )
                results = search_tavily_government(extract_subject(question))
                search_context = results.get("answer", "")
                search_results = results.get("results", [])
                # Ajout du contenu des résultats pour enrichir le contexte
                if search_results:
                    search_context += "\n\n" + "\n\n".join([item.get("content", "") for item in search_results])

            elif st.session_state.get("search_engine") == "Google":
                status_placeholder.markdown(
                    '<div class="status-message">🌐 Recherche internet (Google)...</div>',
                    unsafe_allow_html=True
                )
                results = search_google_government(extract_subject(question))
                search_results = results.get("results", [])
                # Google renvoie "snippet"
                search_context = "\n\n".join([item.get("snippet", "") for item in search_results])

        # Étape 5 : Génération de la réponse
        status_placeholder.markdown(
            '<div class="status-message">🤖 Génération de la réponse par Mistral...</div>',
            unsafe_allow_html=True
        )
        prompt = build_parlementary_response_prompt(
            question=question,
            parliamentary_context=parliamentary_context,
            legal_context=legal_context,
            uploaded_documents=uploaded_docs_context,
            detail_juridique=detail_juridique,
            longueur=longueur,
            response_orientation=response_orientation,
            custom_instructions=custom_instructions,
            search_context=search_context
        )
        mistral_response = call_mistral_parlementary_response(
            prompt,
            longueur,
            question,
            model_size=model_size
        )

        # ➡️ Effacer le message
        status_placeholder.empty()

        return {
            "question": question,
            "context": [doc.reponse for doc in similar_documents[:6] if doc.reponse],
            "context_str": parliamentary_context,
            "legal_context": legal_context,
            "response": mistral_response,
            "legal_sources": legal_sources if include_legal_articles else [],
            "similar_documents": similar_documents,
            "uploaded_documents": uploaded_results,
            "search_results": search_results,
            "metadata": {
                "status": "success",
                "model_used": f"mistral-{model_size}-latest",
                "timestamp": datetime.now(pytz.timezone('Europe/Paris')).isoformat(),
                "legislature": legislature,
                "rubrique": rubrique
            }
        }

    except Exception as e:
        return {
            "question": question,
            "response": f"Erreur lors de la génération de la réponse : {str(e)}",
            "error": str(e),
            "legal_sources": [],
            "similar_documents": [],
            "metadata": {
                "status": "error",
                "timestamp": datetime.now(pytz.timezone('Europe/Paris')).isoformat(),
                "legislature": legislature,
                "rubrique": rubrique
            }
        }

# --- NOUVEL ENDPOINT SIMPLIFIÉ POUR LISTER LES DOCUMENTS ---

# Fonction qui retourne juste les deux catégories de documents - VERSION STATIQUE
def get_simple_documents_list():
    return {
        "documents": [
            {
                "type": "Questions écrites (QE) de l'Assemblée nationale",
                "periode": "2017-2025",
                "description": "Questions ayant obtenu une réponse ministérielle (avant le 1er novembre 2025)."
            },
            {
                "type": "Questions écrites (QE) du Sénat",
                "periode": "2017-2025",
                "description": "Collection complète des questions écrites ayant obtenu une réponse ministérielle (avant le 1er novembre 2025)."
            }
        ]
    }

#################################################################
### -------------  2. AUTHENTIFICATION -----------------------###
#################################################################

config = {
    "credentials": {
        "usernames": {
            "Whisler": {
                "name": "Francois-Mathieu",
                "password": os.getenv("USER_WHISLER_PASSWORD")
            },
            "Delphine": {
                "name": "Caudilla Delphine",
                "password": os.getenv("USER_DELPHINE_PASSWORD")
            },
            "Isabelle": {
                "name": "Caudilla Isabelle",
                "password": os.getenv("USER_ISABELLE_PASSWORD")
            }
        }
    }
}

if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None

def check_password():
    if st.session_state["username"] in config['credentials']['usernames']:
        stored_password = config['credentials']['usernames'][st.session_state["username"]]["password"]
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == stored_password:
            st.session_state["authentication_status"] = True
            st.session_state["name"] = config['credentials']['usernames'][st.session_state["username"]]["name"]
            return
    st.session_state["authentication_status"] = False

# --- Connexion ou contenu principal ---

if st.session_state.authentication_status is not True:
    # Masquer la sidebar avant connexion
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        /* Centrer le titre */
        .auth-title {text-align: center; margin-top: 0.5rem;}
        /* Centrer le paragraphe d'intro */
        .auth-intro {text-align: center; color: #5c5c5c;}
        </style>
    """, unsafe_allow_html=True)

    # --- Page d'authentification ---
    st.markdown('<h1 class="auth-title">🔐 Authentification requise</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-intro">Veuillez entrer vos identifiants pour accéder au générateur de réponses aux questions écrites.</p>', unsafe_allow_html=True)

    # Colonnes pour réduire la largeur et centrer les champs
    # Ajuste les ratios pour obtenir la largeur souhaitée (ici ~25% de la page)
    left, center, right = st.columns([3, 2, 3])
    with center:
        st.text_input("Nom d'utilisateur", key="username")
        st.text_input("Mot de passe", type="password", key="password")
        if st.button("Se connecter"):
            check_password()
            st.rerun()

    if st.session_state.authentication_status is False:
        st.error("Identifiants incorrects. Veuillez réessayer.")



else:
    # Réafficher la sidebar après connexion

    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: block;}
        </style>
    """, unsafe_allow_html=True)

#################################################################
### ---- 3. AFFICHAGE DU SITE APRES CONNEXION --------------- ###
#################################################################

    # --- Initialisation de l'historique (UNIQUEMENT si non existant) ---
    if "full_historique" not in st.session_state:
        st.session_state.full_historique = {}
        save_historique()  # Sauvegarde initiale

    # --- Chargement de l'historique depuis session_state (si vide, on vérifie le cache) ---
    if not st.session_state.full_historique and "historique_cache" in st.session_state:
        st.session_state.full_historique = st.session_state.historique_cache

# --- 6. Configuration de la page et CSS ---
    st.markdown("""
    <style>
        /* Élargir le conteneur principal */
        .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        /* Élargir les zones de saisie */
        textarea, .stTextArea textarea {
            width: 100% !important;
        }
        /* Élargir les selectbox et sliders */
        .stSelectbox, .stSlider {
            width: 100% !important;
        }
        /* Votre CSS existant */
        .stApp { background-color: #f8f9fa; }
        .stTabs [data-baseweb="tab-list"] { gap: 0; background-color: #e9ecef; border-radius: 6px 6px 0 0; padding: 4px; }
        .stTabs [data-baseweb="tab"] { height: 36px; white-space: pre-wrap; background-color: #f8f9fa; border: none; border-radius: 4px 4px 0 0; padding: 0 12px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; font-weight: bold; color: #3d3d3d; }
        .stButton>button { background-color: #4a8bfc; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: 500; }
        .stButton>button:hover { background-color: #3a7bfc; }
        .stExpander { background-color: #ffffff; border: 1px solid #e9ecef; border-radius: 6px; margin-bottom: 0px; } /* ← réduit l’espace */
        .stTextArea textarea { font-family: 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.5; }
        .stAlert { border-radius: 6px; }
        .source-text { font-family: monospace; font-size: 14px; background-color: #f8f9fa; padding: 8px; border-radius: 4px; border-left: 3px solid #4a8bfc; }
        .response-text { font-family: 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.6; white-space: pre-wrap; background-color: white; padding: 16px; border-radius: 6px; border: 1px solid #e9ecef; }

        /* Nouveau : titre de l'historique */
        .history-title {
            margin-bottom: 15px;
        }

        /* Style carte avec ombre légère pour les expanders */
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 5px !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); /* ombre douce */
        }

    </style>
    """, unsafe_allow_html=True)

    st.title("🏛️ Générateur de réponses aux questions écrites parlementaires")
    st.markdown("""
    Application (version Beta) de réponse aux questions parlementaires,
    appuyée sur une base documentaire (embedding avec **camemBERT**), un moteur de recherche (**Tavily** ou **Google**) et le modèle **Mistral**.
    """)

#################################################################
#### -------------------- 3a. SIDEBAR  --------------------- ####
#################################################################

    with st.sidebar:

        # Bouton Déconnexion
        if st.button('Déconnexion', key="logout"):
            st.session_state.authentication_status = None
            st.rerun()

        # Message de bienvenue
        st.write(f'Bienvenue *{st.session_state["name"]}*')
    
        # --- Version du site ---
        st.markdown(
            """
            <style>
            .version-text {
                position: absolute;
                top: 5px;
                left: 0px;
                color: #555;
                font-size: 14px;
                font-style: italic;
            }
            </style>
            <div class="version-text">Version v0.1.12 (Beta)</div>
            """,
            unsafe_allow_html=True
        )   

        # Séparateur visuel
        st.markdown("---")

        # Paramètres de mode
        mode = st.radio(
            "Type de réponse souhaitée",
            ["Réponse parlementaire"], # Ajouter , "Analyse juridique" pour avoir le second mode - Permet de moduler les modes accessibles sur le site
            index=0
        )

        # Conteneur pour contrôler la largeur du bouton
        button_container = st.container()
        # Initialisation des bouton à False
        generate_parliamentary_button = False
        generate_analysis_button = False
        with button_container:
            if mode == "Réponse parlementaire":
                generate_parliamentary_button = st.button(
                    "Générer la réponse",
                    type="primary",
                    key="generate_parliamentary_button"
                    # Sans use_container_width pour une largeur automatique
                )
            elif mode == "Analyse juridique":
                generate_analysis_button = st.button(
                    "Générer l'analyse",
                    type="primary",
                    key="generate_analysis_button"
                    # Sans use_container_width pour une largeur automatique
                )
        
        # Ajoutez une séparation visuelle supplémentaire
        st.markdown("---")

        # Choix du modèle Mistral
        model_size = st.radio(
            "Choix du modèle Mistral",
            ["Small", "Medium", "Large (recommandé)"],
            index=0
        )

        # Normaliser la valeur pour l'appel API
        model_size = model_size.lower()

        # Choix du moteur de recherche
        search_engine = st.radio(
            "Moteur de recherche internet",
            ["Google", "Tavily"],
            index=0
        )
        st.session_state["search_engine"] = search_engine

        # Séparateur
        st.markdown("---")

        # Case à cocher pour Gérer la base documentaire
        if 'manage_doc_base' not in st.session_state:
            st.session_state.manage_doc_base = False

        manage_doc_base = st.checkbox(
            "Gérer la base documentaire",
            value=st.session_state.manage_doc_base,
            key="manage_doc_base_checkbox",
            help="Active l'interface pour ajouter/supprimer des documents dans le RAG"
        )

##########################################################################
### ------ 3b. INTERFACE DE GESTION DE LA BASE DOCUMENTAIRE ---------- ###
##########################################################################

    # Initialisation du bouton search (A REPOSITIONNER)
    search_button = False
    # Affichage conditionnel de l'interface de gestion de la base documentaire
    if manage_doc_base:
        st.session_state.manage_doc_base = True  # Met à jour l'état
        st.markdown("---")
        st.markdown("#### 📚 Gestion de la base documentaire")

        # 1. Liste des documents (collections)
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            protected_collections = {
                "Code de la sécurité sociale",
                "Code du travail",
                "CASF",
                "QuestionParlementaire",
                "Code de la santé publique"
            }

            # Filtre pour ne garder que les collections "documents" (ex: "NomDuDocument_2023")
            doc_collections = [
                name for name in collection_names
                if name not in protected_collections and "_" in name  # Ex: "MonDocument_2023"
            ]

            if not doc_collections:
                st.info("Aucun document trouvé.")
            else:
                st.markdown("**Documents disponibles :**")

                # Tri alphabétique sur le nom nettoyé
                sorted_collections = sorted(
                    doc_collections,
                    key=lambda name: name.split('__')[0].replace('_', ' ').lower()
                )

                for doc_name in sorted_collections:
                    clean_name = doc_name.split('__')[0].replace('_', ' ')
                    col1, col2, col3 = st.columns([18, 1, 1])  # plus de place pour le nom

                    with col1:
                        st.markdown(f"📄 {clean_name}")

                    with col2:
                        if st.button("✏️", key=f"rename_{doc_name}", help="Renommer le document"):
                            st.session_state.show_rename_modal = True
                            st.session_state.current_doc_to_rename = doc_name
                            st.rerun()

                    with col3:
                        if st.button("🗑️", key=f"del_{doc_name}", help="Supprimer le document"):
                            qdrant_client.delete_collection(collection_name=doc_name)
                            st.success(f"Document '{clean_name}' supprimé.")
                            st.rerun()

                # Fenêtre modale de renommage
                if st.session_state.show_rename_modal:
                    doc_name = st.session_state.current_doc_to_rename
                    clean_name = doc_name.split('__')[0].replace('_', ' ')

                    with st.container():
                        st.markdown("---")
                        st.subheader(f"Renommer le document ''{clean_name}''")

                        new_name = st.text_input(
                            "Nouveau nom:",
                            value=clean_name.replace(" ", "_"),
                            key=f"new_name_{doc_name}"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("Valider", key=f"validate_rename_{doc_name}"):
                                if new_name.strip():
                                    try:
                                        # 1. Vérification si le nouveau nom existe déjà
                                        new_display_name = new_name.strip().replace(" ", "_")
                                        new_collection_name = f"{new_display_name}__{doc_name.split('__')[1]}"

                                        # Récupère toutes les collections existantes
                                        existing_collections = qdrant_client.get_collections()
                                        existing_names = [col.name for col in existing_collections.collections]

                                        # Vérifie si le nouveau nom existe déjà
                                        if new_collection_name in existing_names:
                                            st.error(f"❌ Le nom '{new_name}' existe déjà. Veuillez choisir un autre nom.")
                                        else:
                                            # Initialisation de la progression
                                            progress_bar = st.progress(0)
                                            status_text = st.empty()
                                            estimated_total = 1000  # Estimation conservatrice du nombre total de points

                                            # 2. Crée la nouvelle collection (0-10%)
                                            status_text.text("Création de la nouvelle collection...")
                                            qdrant_client.create_collection(
                                                collection_name=new_collection_name,
                                                vectors_config=models.VectorParams(
                                                    size=1024,
                                                    distance=models.Distance.COSINE
                                                )
                                            )
                                            progress_bar.progress(10)

                                            # 3. Récupère les points (10-40%)
                                            status_text.text("Récupération des données...")
                                            offset = None
                                            all_points = []

                                            while True:
                                                records, offset = qdrant_client.scroll(
                                                    collection_name=doc_name,
                                                    limit=100,
                                                    offset=offset,
                                                    with_payload=True,
                                                    with_vectors=True
                                                )

                                                for record in records:
                                                    point_dict = {
                                                        "id": str(record.id),
                                                        "vector": record.vector,
                                                        "payload": record.payload
                                                    }
                                                    all_points.append(models.PointStruct(**point_dict))

                                                if offset is None:
                                                    break

                                                # Calcul sécurisé de la progression (max 40%)
                                                current_progress = min(40, 10 + int(30 * len(all_points) / max(1, estimated_total)))
                                                progress_bar.progress(current_progress)

                                            # 4. Copie les points (40-90%)
                                            status_text.text(f"Copie des {len(all_points)} chunks...")
                                            batch_size = 50

                                            for i in range(0, len(all_points), batch_size):
                                                batch = all_points[i:i + batch_size]
                                                qdrant_client.upsert(
                                                    collection_name=new_collection_name,
                                                    points=batch,
                                                    wait=True
                                                )

                                                # Calcul SÉCURISÉ de la progression (max 90%)
                                                batch_progress = min(90, 40 + int(50 * (i + len(batch)) / max(1, len(all_points))))
                                                progress_bar.progress(batch_progress)

                                            # 5. Supprime l'ancienne collection (100%)
                                            status_text.text("Finalisation...")
                                            qdrant_client.delete_collection(collection_name=doc_name)
                                            progress_bar.progress(100)

                                            st.success(f"✅ Document renommé en '{new_name}' !")
                                            st.session_state.show_rename_modal = False
                                            st.rerun()

                                    except Exception as e:
                                        st.error(f"Erreur: {e}")
                                else:
                                    st.warning("Veuillez entrer un nom valide.")


                        with col2:
                            if st.button("Annuler", key=f"cancel_rename_{doc_name}"):
                                st.session_state.show_rename_modal = False
                                st.rerun()

        except Exception as e:
            st.error(f"Erreur: {e}")

        # 2. Section d'upload de documents
        st.markdown("---")
        st.markdown("**Ajouter un document**")

        uploaded_file = st.file_uploader(
            "Sélectionnez un PDF ou Word",
            type=["pdf", "docx"],
            key="doc_uploader"
        )

        if uploaded_file:
            # Champ pour le nom personnalisé (sera aussi le nom de la collection)
            default_name = os.path.splitext(uploaded_file.name)[0]
            custom_name = st.text_input(
                "Nom du document (sera aussi le nom de la collection):",
                value=default_name,
                key="doc_name_input"
            )

            if st.button("Ajouter le document", key="add_document"):
                if not custom_name.strip():
                    st.warning("Veuillez entrer un nom valide.")
                else:
                    # Génération du nom de la collection (avec timestamp caché pour l'unicité)
                    display_name = custom_name.strip().replace(" ", "_")  # Nom affiché (sans timestamp)
                    collection_name = f"{display_name}__{int(datetime.now().timestamp())}"  # Nom interne (avec timestamp)

                    # Initialisation de la barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current, total, message):
                        """Met à jour la barre de progression et le statut."""
                        percent = int((current / total) * 100) if total > 0 else 0
                        progress_bar.progress(percent)
                        status_text.text(f"{message} ({current}/{total})")

                    try:
                        # Création de la collection
                        status_text.text("Création de la collection dans Qdrant...")
                        qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=models.VectorParams(
                                size=1024,
                                distance=models.Distance.COSINE
                            )
                        )

                        # Sauvegarde du fichier temporaire
                        status_text.text("Sauvegarde du fichier temporaire...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # Appel de la fonction avec callback
                        status_text.text("Traitement et indexation en cours...")
                        success = process_and_index_document(
                            file_path=tmp_path,
                            file_type=uploaded_file.name.split('.')[-1],
                            collection_name=collection_name,
                            qdrant_client=qdrant_client,
                            embedding_model=embedding_model,
                            progress_callback=update_progress  # ← Callback ajouté
                        )

                        if success:
                            status_text.text("Document ajouté avec succès !")
                            progress_bar.progress(100)
                            st.success(f"✅ Document ajouté sous le nom '{custom_name}' !")

                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                            st.session_state.manage_doc_base = False
                            st.rerun()
                        else:
                            status_text.text("Échec de l'ajout du document")
                            st.error("❌ Échec de l'ajout.")

                    except Exception as e:
                        status_text.text(f"Erreur: {str(e)}")
                        st.error(f"Erreur: {e}")
                    finally:
                        time.sleep(2)  # Laisse le temps de voir le résultat
                        progress_bar.empty()
                        status_text.empty()

#########################################################################################
#### ---------- 3c. INTERFACE DE REPONSE AUX QUESTIONS PARLEMENTAIRES -------------- ####
#########################################################################################

    else:
        st.markdown("#### Question parlementaire")
        question = st.text_area(
            "Question parlementaire",   # libellé non vide
            height=150,
            placeholder="Copier ici le texte de la question parlementaire",
            key="question_input",
            label_visibility="collapsed"
        )

        # Initialisation des variables des boutons (pour éviter NameError)
        search_button = False

        # --- Boutons conditionnels selon le mode ---
        if mode == "Réponse parlementaire":
            # --- Paramètres de réponse ---
            st.markdown("#### Paramètres de réponse")
            col1, col2, col3 = st.columns(3)
            with col1:
                response_orientation_options = [
                    "Répondre de façon neutre",
                    "Répondre négativement aux propositions du parlementaire",
                    "Répondre positivement aux propositions du parlementaire",
                    "Répondre de manière technique et détaillée"
                ]
                selected_orientation = st.selectbox(
                    "Orientation de la réponse",
                    response_orientation_options,
                    index=0
                )
            with col2:
                longueur = st.selectbox(
                    "Longueur de la réponse",
                    ["Courte (300 mots)", "Moyenne (500 mots)", "Longue (1000 mots)"],
                    index=1
                )
            with col3:
                include_legal_articles = st.selectbox(
                    "Inclure une recherche dans les codes juridiques",
                    ["Non", "Oui"],
                    index=0,
                    key="include_legal_articles_select"
                ) == "Oui"

            MAX_LEN = 300

            # Ligne avec deux colonnes : champ texte + case à cocher
            colA, colB = st.columns([2, 1])

            with colA:
                MAX_LEN = 300
                custom_instructions = st.text_area(
                    "Optionnel : instructions supplémentaires pour la réponse (max. 300 caractères)",
                    placeholder="Ex: Insister sur l'aspect budgétaire, mentionner le projet de loi X...",
                    height=100,
                    key="custom_instructions"
                )
                if custom_instructions:
                    remaining = MAX_LEN - len(custom_instructions)
                    if remaining < 0:
                        st.warning(f"⚠️ Vous avez dépassé la limite de {MAX_LEN} caractères ({len(custom_instructions)} actuellement).")
                        custom_instructions = custom_instructions[:MAX_LEN]

            with colB:
                detail_juridique = st.slider(
                    "Niveau de détail juridique (1 = bas, 5 = élevé)",
                    min_value=1,
                    max_value=5,
                    value=3
                )

                if include_legal_articles:
                    must_contain = st.text_input(
                        "Optionnel : les articles sélectionnés doivent contenir (mot ou expression exacte)",
                        key="must_contain_input",
                        placeholder="Ex: allocation, article 123, décret 2020-..."
                    )
                else:
                    must_contain = ""

            st.markdown("**Limiter les documents sources** (attention : la réponse n'intègre ni plus les anciennes QE, ni les textes juridiques et ni la recherche internet)")
            use_priority_docs = st.checkbox(
                "Rechercher uniquement dans les documents suivants :",
                value=False,
                key="use_priority_docs"
            )

            if use_priority_docs:
                # Récupère les collections "documents"
                collections = qdrant_client.get_collections()
                doc_collections = [
                    col.name for col in collections.collections
                    if col.name not in {"Code_de_la_sécurité_sociale", "Code_du_travail", "CASF", "QuestionParlementaire", "Code_de_la_santé_publique"}
                    and "_" in col.name
                ]

                # Affiche les noms propres (sans timestamp)
                doc_names = [col.split('__')[0].replace('_', ' ') for col in doc_collections]
                selected_docs = st.multiselect(
                    "Sélectionnez les documents",
                    options=doc_names,
                    key="priority_docs",
                    placeholder="Choisir..."
                )

        elif mode == "Analyse juridique":
            # --- Paramètres de recherche juridique ---
            st.markdown("### Paramètres de recherche juridique")
            col1, col2, col3 = st.columns(3)
            with col1:
                must_contain = st.text_input("🔎 Doit contenir (mot ou expression exacte)", key="must_contain_input")
            with col2:
                threshold = st.selectbox(
                    "📊 Seuil de sélection",
                    [0.4, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85],
                    index=2,
                    key="threshold_select"
                )
            with col3:
                max_articles = st.number_input(
                    "📌 Nombre maximum d'articles",
                    min_value=1,
                    max_value=30,
                    value=5,
                    key="max_articles_input"
                )

            # --- Affichage du bouton d'analyse juridique ---
            search_button = st.button("Rechercher les articles", type="secondary", key="search_button")

###################################################################
#### --------------- 4. GENERATION DE LA REPONSE ------------- ####
###################################################################

    if generate_parliamentary_button or search_button or generate_analysis_button:
        if not question.strip():
            st.warning("Veuillez entrer une question.")
        else:
            try:
                debug_logs = ""
                response_data = {}
                stats = {}

                if mode == "Réponse parlementaire":
                    # Logique inchangée pour le mode "Réponse parlementaire"
                    response_data = generate_response(
                        question=question,
                        detail_juridique=detail_juridique,
                        longueur=longueur,
                        response_orientation=selected_orientation,
                        custom_instructions=custom_instructions,
                        include_legal_articles=include_legal_articles,
                        must_contain=must_contain if include_legal_articles else "",
                        max_legal_articles=detail_juridique # le nombre d'articles est égal au niveau de détail juridique
                    )

                elif mode == "Analyse juridique":
                    response_data = generate_legal_analysis(
                        question=question,
                        must_contain=must_contain,
                        max_articles=max_articles,
                        threshold=threshold,
                        model_size=model_size,
                        search_button=search_button,  # Bouton "Rechercher les articles"
                        generate_analysis_button=generate_analysis_button  # Bouton "Générer l'analyse"
                    )

                # Détermination des onglets à afficher
                tabs = []
                if mode == "Réponse parlementaire":
                    tabs = ["📜 Réponse"]
                    if include_legal_articles:
                        tabs.append("⚖️ Articles juridiques")
                    tabs.extend(["🏛️ Anciennes QE", "📰 Recherches actualités", "📄 Base documentaire"])
                elif mode == "Analyse juridique":
                    tabs = ["⚖️ Articles juridiques"]
                    if generate_analysis_button and response_data.get("response"):
                        tabs.insert(0, "📜 Analyse")  # Ajoute "Analyse" en premier si générée

                # Création dynamique des onglets
                if tabs:
                    st_tabs = st.tabs(tabs)

                    # Affichage du contenu en fonction des onglets
                    for i, tab in enumerate(st_tabs):
                        with tab:
                            if mode == "Réponse parlementaire":
                                if "📜 Réponse" in tabs[i]:
                                    st.markdown("#### Réponse générée")
                                    st.markdown(response_data["response"])
                                    if response_data.get("debug_logs"):
                                        with st.expander("🐛 Voir les logs de recherche"):
                                            st.text_area("Logs", response_data["debug_logs"], height=200)
                                    # Bouton d'export
                                    if response_data.get("response"):
                                        export_content = build_export_content(
                                            response_data,
                                            mode="parlementaire",  # ← Mode codé en dur (valide car dans le bloc "Réponse parlementaire")
                                            include_legal_articles=include_legal_articles  # ← Utilise la variable existante
                                        )
                                        st.download_button(
                                            label="📥 Exporter en TXT",
                                            data=export_content.encode("utf-8"),
                                            file_name="export_reponse_parlementaire.txt",  # ← Nom de fichier plus clair
                                            mime="text/plain",
                                            key=f"export_reponse_{i}"  # ← Clé unique basée sur l'index
                                        )
                                    st.markdown('<div style="height: 300px;"></div>', unsafe_allow_html=True)

                                elif "🏛️ Anciennes QE" in tabs[i]:
                                    st.markdown("#### 5 Questions parlementaires les plus similaires de la plus récente à la plus ancienne")
                                    if not response_data.get("similar_documents"):
                                        st.info("Aucune question parlementaire similaire trouvée.")
                                    else:
                                        similar_documents = response_data["similar_documents"]

                                        # Tri par date décroissante
                                        similar_documents = sorted(
                                            similar_documents,
                                            key=lambda d: safe_parse_date(d.date_reponse),
                                            reverse=True
                                        )

                                        # Affichage avec score
                                        for idx, doc in enumerate(similar_documents):
                                            chambre = doc.chambre or ("Assemblée nationale" if str(doc.uid).startswith("QAN") else "Sénat")
                                            date_reponse = doc.date_reponse or "Inconnue"
                                            question_text = doc.question or "Question non disponible"
                                            reponse_text = doc.reponse or "Réponse non disponible"
                                            score = f"{doc.score:.2f}" if doc.score is not None else "N/A"

                                            with st.expander(f"{idx+1}. QE {doc.uid} ({chambre}) - Score de proximité : {score}"):
                                                st.markdown(f"**Date de réponse:** {date_reponse}")
                                                st.markdown(f"**Chambre:** {chambre}")
                                                st.markdown(f"**Question:** {question_text}")
                                                st.markdown(f"**Réponse:** {reponse_text}")

                                    st.markdown('<div style="height: 300px;"></div>', unsafe_allow_html=True)

                                elif "⚖️ Articles juridiques" in tabs[i]:
                                    st.markdown("#### Articles juridiques pertinents")

                                    legal_sources = response_data.get("legal_sources", [])

                                    if not legal_sources:
                                        st.info("Aucun texte juridique cité.")
                                    else:
                                        for idx, art in enumerate(legal_sources):
                                            score = f"{art.score:.2f}" if art.score is not None else "N/A"
                                            with st.expander(f"{idx+1}. Article {art.num} ({art.collection}) - Score : {score}"):
                                                st.markdown(f"**Titre:** {art.titre}")
                                                st.markdown(f"**Contexte hiérarchique:** {art.contexte_hierarchique}")
                                                st.markdown(f"**Texte complet:**\n\n{art.article_complet}")

                                elif "📰 Recherches actualités" in tabs[i]:
                                    st.markdown("#### Dernières annonces et actualités gouvernementales")
                                    search_results = response_data.get("search_results", [])
                                    if not search_results:
                                        st.info("Aucune actualité trouvée via le moteur de recherche.")
                                    else:
                                        for idx, item in enumerate(search_results):
                                            titre = item.get("title", "Sans titre")
                                            url = item.get("url", "")
                                            # Utiliser "content" pour Tavily, "snippet" pour Google
                                            extrait = item.get("content") or item.get("snippet") or ""
                                            with st.expander(f"{idx+1}. {titre}"):
                                                if url:
                                                    st.markdown(f"[Lien vers la source]({url})")
                                                if extrait:
                                                    st.markdown(extrait)

                                elif "📄 Base documentaire" in tabs[i]:
                                    st.markdown("#### Résultats pertinents dans les documents uploadés")

                                    uploaded_results = response_data.get("uploaded_documents", [])

                                    # Filtrer par score
                                    min_score = 0.7
                                    filtered_results = [res for res in uploaded_results if res.get("score", 0) >= min_score]

                                    if not filtered_results:
                                        st.info("Aucun extrait pertinent trouvé dans les documents uploadés.")
                                    else:
                                        # Regrouper par document
                                        grouped = {}
                                        for res in filtered_results:
                                            doc_name = res["collection"].split('__')[0].replace('_', ' ')
                                            grouped.setdefault(doc_name, []).append(res)

                                        for doc_name, results in grouped.items():
                                            with st.expander(f"📄 {doc_name} ({len(results)} extraits)"):
                                                for idx, res in enumerate(results, start=1):
                                                    score = f"{res['score']:.2f}" if res.get("score") is not None else "N/A"
                                                    text_preview = res["text"]
                                                    title = res.get("title") or "N/A"

                                                    st.markdown(f"**Extrait {idx} (score: {score})**")
                                                    st.markdown(text_preview)
                                                    st.markdown(f"**Section :** {title}")
                                                    st.markdown("---")

                                                # Actions sur la collection
                                                col1, col2 = st.columns([1, 1])
                                                target_collection = results[0]["collection"]
                                                with col1:
                                                    if st.button("✏️ Renommer", key=f"rename_{target_collection}"):
                                                        st.session_state.show_rename_modal = True
                                                        st.session_state.current_doc_to_rename = target_collection
                                                        st.rerun()
                                                with col2:
                                                    if st.button("🗑️ Supprimer", key=f"del_{target_collection}"):
                                                        qdrant_client.delete_collection(collection_name=target_collection)
                                                        st.success(f"Document '{doc_name}' supprimé.")
                                                        st.rerun()

                            elif mode == "Analyse juridique":
                                if "📜 Analyse" in tabs[i]:
                                    st.markdown("#### Analyse juridique générée")
                                    st.markdown(response_data["response"])

                                    # Affichage des logs (si disponibles)
                                    if response_data.get("debug_logs"):
                                        with st.expander("🐛 Voir les logs de recherche"):
                                            st.text_area("Logs", response_data["debug_logs"], height=200)

                                    # Bouton d'export
                                    if response_data.get("response"):
                                        export_content = build_export_content(
                                            response_data,
                                            mode="analyse",  # Mode pour l'export
                                            include_legal_articles=False
                                        )
                                        st.download_button(
                                            label="📥 Exporter en TXT",
                                            data=export_content.encode("utf-8"),
                                            file_name=f"analyse_juridique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",  # Nom de fichier unique
                                            mime="text/plain",
                                            key=f"export_analyse_{i}"
                                        )
                                    st.markdown('<div style="height: 300px;"></div>', unsafe_allow_html=True)

                                elif "⚖️ Articles juridiques" in tabs[i]:
                                    st.markdown("#### Articles juridiques pertinents")

                                    # Vérification de la présence de sources
                                    if not response_data.get("sources"):
                                        st.info("Aucun article juridique trouvé.")
                                    else:
                                        # Construction de l'arbre législatif (comme dans ton code original)
                                        tree = {}
                                        for source in response_data["sources"]:
                                            art = source["article"]  # Objet RetrievedLegalDocument
                                            add_to_tree(tree, art, source)  # Ajoute l'article à l'arbre

                                        # Affichage de l'arbre avec render_tree
                                        render_tree(st, tree)


            except Exception as e:
                st.error(f"Erreur inattendue: {str(e)}")
                st.exception(e)

###################################################################
#### -------- 5. GESTION ET AFFICHAGE DE L'HISTORIQUE -------- ####
###################################################################

    # Initialisation du mode
    mode_value = mode.lower().replace(" ", "_")  # "réponse_parlementaire" ou "analyse_juridique"
    # Ajout d'une entrée à l'historique après génération d'une réponse
    if generate_parliamentary_button or generate_analysis_button:
        if not question.strip():
            st.warning("Veuillez entrer une question.")
        else:
            try:
                # Génération d'une clé unique pour la question
                q_hash = hashlib.md5(question.encode()).hexdigest()
                # Préparation des métadonnées communes
                metadata = {
                    "mode": mode_value,
                    "timestamp": datetime.now(pytz.timezone('Europe/Paris')).isoformat(),
                    "model_used": f"mistral-{model_size}-latest",
                    "include_legal_articles": include_legal_articles if mode_value == "parlementaire" else False,
                    "legislature": None,
                    "rubrique": None
                }

                # Filtrer les documents uploadés par score (seuil = 0.7)
                min_score = 0.7
                uploaded_docs_filtered = [
                    res for res in response_data.get("uploaded_documents", [])
                    if res.get("score", 0) >= min_score
                ]

                # Ajout à l'historique
                st.session_state.full_historique[q_hash] = {
                    "question": question,
                    "response": response_data.get("response", "Pas de réponse générée"),
                    "similar_documents": response_data.get("similar_documents", []),
                    "legal_sources": response_data.get("legal_sources", []),
                    "sources": response_data.get("sources", []),  # Pour le mode "analyse"
                    "search_results": response_data.get("search_results", []),
                    "uploaded_documents": uploaded_docs_filtered,
                    "metadata": metadata
                }
                save_historique()  # Sauvegarde immédiate
            except Exception as e:
                st.error(f"Erreur lors de l'ajout à l'historique: {str(e)}")

    # --- Affichage de l'historique complet ---
    st.markdown("""
    <style>
        /* Centrage du bloc historique */
        .history-container {
            max-width: 1000px;
            margin-left: auto;
            margin-right: 0;
            padding-left: 1rem;
        }
        /* Style des expanders */
        .history-expander {
            margin-bottom: 0.5rem !important;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }
        /* Espacement des éléments */
        .history-entry {
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Conteneur pour centrer à gauche
    with st.container():
        st.markdown('<div class="history-container">', unsafe_allow_html=True)

        if hasattr(st.session_state, 'full_historique') and st.session_state.full_historique:
            nb_entries = len(st.session_state.full_historique)
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
                    <span style="font-size:18px;font-weight:bold;">🗂️ Historique des questions ({nb_entries})</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Tri du plus récent au plus ancien
            sorted_historique = sorted(
                st.session_state.full_historique.items(),
                key=lambda x: x[1]["metadata"].get("timestamp", ""),
                reverse=True
            )

            for idx, (q_hash, entry) in enumerate(sorted_historique):
                with st.expander(f"{idx+1}. {truncate_text(entry['question'], max_tokens=50)}", expanded=False):
                    # Métadonnées
                    metadata = entry.get("metadata", {})
                    st.caption(
                        f"🕒 {metadata.get('timestamp', '')[:16].replace('T', ' ')} "
                        f"| Mode: {metadata.get('mode', 'inconnu')} "
                        f"| Modèle: {metadata.get('model_used', 'inconnu')}"
                    )

                    # Question
                    st.markdown(f"**📝 Question:**\n{entry.get('question', 'Non disponible')}")

                    # Réponse (sans duplication)
                    st.markdown(f"**💬 Réponse:**")
                    st.markdown(entry.get('response', 'Non disponible'))

                    # --- Sources juridiques (si disponibles) ---
                    if entry.get("legal_sources") or entry.get("sources"):
                        with st.expander("⚖️ Articles juridiques"):
                            if metadata.get("mode") == "analyse_juridique":  # Note : "analyse_juridique" (avec underscore)
                                # Mode "Analyse juridique" : utilise "sources" et affiche l'arbre législatif
                                if not entry.get("sources"):
                                    st.info("Aucun article juridique enregistré.")
                                else:
                                    # Construction de l'arbre législatif
                                    tree = {}
                                    for source in entry["sources"]:
                                        art = source["article"]
                                        add_to_tree(tree, art, source)  # Utilise la fonction existante

                                    # Affichage de l'arbre
                                    render_tree(st, tree)  # Utilise la fonction existante

                            else:  # Mode "Réponse parlementaire" : utilise "legal_sources"
                                for art in entry.get("legal_sources", []):
                                    with st.expander(f"Article {getattr(art, 'num', 'N/A')} ({getattr(art, 'collection', 'N/A')})"):
                                        st.markdown(f"**Titre:** {getattr(art, 'titre', 'Non disponible')}")
                                        st.markdown(f"**Texte:**\n{getattr(art, 'article_complet', 'Non disponible')}")

                    # --- Résultats de recherche internet (si disponibles) ---
                    if entry.get("search_results"):
                        with st.expander("🌐 Résultats de recherche internet"):
                            for item_idx, item in enumerate(entry["search_results"]):
                                with st.expander(f"{item_idx+1}. {item.get('title', 'Sans titre')}"):
                                    if item.get("url"):
                                        st.markdown(f"[Lien]({item.get('url')})")
                                    st.markdown(item.get("content") or item.get("snippet", "Aucun extrait disponible"))

                    # --- Anciennes QE similaires (si disponibles) ---
                    if entry.get("similar_documents"):
                        with st.expander("🏛️ Questions parlementaires similaires"):
                            for doc_idx, doc in enumerate(entry["similar_documents"]):
                                with st.expander(f"QE {doc_idx+1} - Score: {getattr(doc, 'score', 'N/A'):.2f}"):
                                    st.markdown(f"**Question:** {getattr(doc, 'question', 'Non disponible')}")
                                    st.markdown(f"**Réponse:** {getattr(doc, 'reponse', 'Non disponible')}")

                    # --- Résultats vectoriels sur documents uploadés ---
                    if entry.get("uploaded_documents"):
                        with st.expander("📄 Documents uploadés pertinents"):
                            # Regrouper par document
                            grouped = {}
                            for res in entry["uploaded_documents"]:
                                doc_name = res["collection"].split('__')[0].replace('_', ' ')
                                grouped.setdefault(doc_name, []).append(res)

                            for doc_name, results in grouped.items():
                                with st.expander(f"📄 {doc_name} ({len(results)} extraits)"):
                                    for idx_res, res in enumerate(results, start=1):
                                        score = f"{res['score']:.2f}" if res.get("score") is not None else "N/A"
                                        text_full = res["text"]  # ✅ affichage complet
                                        title = res.get("title") or "N/A"

                                        st.markdown(f"**Extrait {idx_res} (score: {score})**")
                                        st.markdown(text_full)
                                        st.markdown(f"**Section :** {title}")
                                        st.markdown("---")

                    # Boutons d'action
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        export_content = build_export_content(
                            entry,
                            mode=metadata.get("mode", "parlementaire"),
                            include_legal_articles=metadata.get("include_legal_articles", False)
                        )
                        st.download_button(
                            label="⬇️ Exporter en TXT",
                            data=export_content.encode("utf-8"),
                            file_name=f"export_{idx+1}_{metadata.get('mode', 'parlementaire')}.txt",
                            mime="text/plain",
                            key=f"export_hist_{q_hash}"
                        )
                    with col2:
                        if st.button("🗑️ Supprimer", key=f"del_hist_{q_hash}"):
                            del st.session_state.full_historique[q_hash]
                            save_historique()
                            st.rerun()

        else:
            st.info("Aucune question enregistrée dans l'historique pour le moment.")

        st.markdown('</div>', unsafe_allow_html=True)
