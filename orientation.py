import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. VÉRITÉ ABSOLUE (LA LISTE FORCÉE) ---
# On définit ceci au tout début pour l'injecter partout
CONSTANTE_FILIERES = """
CRITIQUE - TU DOIS RESPECTER STRICTEMENT CETTE LISTE.
L'ENSA Tanger compte EXACTEMENT ces 6 Filières Ingénieur (Cycle Ingénieur) :
1. Génie Systèmes et Réseaux (GSR)
2. Génie Informatique (GINF)
3. Génie Industriel (GIND)
4. Génie des Systèmes Électroniques et Automatiques (GSEA)
5. Génie Énergétique et Environnement Industriel (G2EI)
6. Cybersecurity and Cyberintelligence (CSI)

N'INVENTE JAMAIS D'AUTRE FILIÈRE. SI ON TE PARLE DE "Génie Civil" ou "Mécatronique", dis que cela n'existe pas à l'ENSAT.
"""

# --- 2. CONFIGURATION & SÉCURITÉ ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Clé API non trouvée dans les secrets.")
    st.stop()

st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓", layout="wide")

# --- 3. HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    else:
        st.markdown("# 🏫")
with col2:
    st.title("Assistant Orientation ENSAT")
    st.markdown("**National School of Applied Sciences of Tangier**")

st.divider()

# --- 4. GESTION DE L'ÉTAT (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# --- 5. CHARGEMENT DES DONNÉES (OPTIMISÉ) ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path):
        return None, "Le dossier 'data' n'existe pas."
    
    # Lecture PDF et TXT
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf') or f.endswith('.txt')]
    if not files:
        return None, "Aucun fichier trouvé."

    try:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = loader.load()
            all_docs.extend(docs)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, None
    except Exception as e:
        return None, str(e)

with st.spinner("Chargement de la base de connaissances..."):
    vectorstore, error_msg = initialize_vectorstore()

if error_msg:
    st.error(f"Erreur : {error_msg}")
    st.stop()

# --- 6. MENU SIDEBAR ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    
    if st.button("💬 Chat avec l'IA", use_container_width=True):
        st.session_state.mode = "chat"
    if st.button("📊 Analyseur de Notes", use_container_width=True):
        st.session_state.mode = "grades"
    if st.button("📝 Test Orientation (15 Q)", use_container_width=True):
        st.session_state.mode = "quiz"
    if st.button("⚖️ Comparateur de Filières", use_container_width=True):
        st.session_state.mode = "compare"
    if st.button("🗺️ Roadmap Visuelle", use_container_width=True):
        st.session_state.mode = "roadmap"
        
    st.divider()
    if st.button("🗑️ Reset Historique"):
        st.session_state.messages = []
        st.rerun()

# --- 7. LOGIQUE PRINCIPALE SELON LE MODE ---

# ==========================================
# MODE 1 : ANALYSEUR DE NOTES
# ==========================================
if st.session_state.mode == "grades":
    st.markdown("### 📊 Analyseur de Notes & Compatibilité")
    st.info("L'IA va calculer votre compatibilité avec les 6 filières officielles.")

    with st.form("grade_form"):
        col1, col2 = st.columns(2)
        with col1:
            note_math = st.number_input("Mathématiques (/20)", 0.0, 20.0, 12.0)
            note_phys = st.number_input("Physique / Élec (/20)", 0.0, 20.0, 12.0)
        with col2:
            note_info = st.number_input("Informatique / Algo (/20)", 0.0, 20.0, 12.0)
            note_lang = st.number_input("Français / Anglais (/20)", 0.0, 20.0, 12.0)
            
        note_chimie = st.slider("Aisance en Chimie/Bio", 0, 20, 10)
        
        submitted = st.form_submit_button("📈 Calculer mes Compatibilités")

        if submitted:
            with st.spinner("Calcul des scores..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Prérequis filières matières coefficients")
                context = "\n".join([d.page_content for d in docs])
                
                notes_summary = f"Maths:{note_math}, Phys:{note_phys}, Info:{note_info}, Lang:{note_lang}, Chimie:{note_chimie}"
                
                # INJECTION DE LA LISTE FORCÉE
                prompt = f"""
                Tu es un Analyste Académique de l'ENSA Tanger.
                {CONSTANTE_FILIERES}
                
                MISSION : Calcule un "Score de Compatibilité" (%) pour CHACUNE des 6 filières officielles ci-dessus.
                NOTES ÉTUDIANT : {notes_summary}
                CONTEXTE PDF : {context}
                
                Réponds par un Tableau Markdown.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Analyse Notes :**\n{response.content}"})

# ==========================================
# MODE 2 : ROADMAP VISUELLE
# ==========================================
elif st.session_state.mode == "roadmap":
    st.markdown("### 🗺️ Générateur de Parcours Visuel")
    filiere_cible = st.text_input("Quelle filière visualiser ?", placeholder="Ex: GINF, GSR, CSI...")
    
    if st.button("Générer la Roadmap"):
        if filiere_cible:
            with st.spinner("Dessin..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(f"Programme {filiere_cible}")
                context = "\n".join([d.page_content for d in docs])
                
                graph_prompt = f"""
                Crée un diagramme Graphviz (DOT) pour : {filiere_cible}.
                Contexte: {context}.
                {CONSTANTE_FILIERES}
                Règles : digraph G {{ rankdir=LR; node [shape=box, style=filled, fillcolor=lightblue];
                Nœuds : Année3 -> Année4 -> Ann
