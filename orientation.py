import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION & SÉCURITÉ ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Clé API non trouvée dans les secrets.")
    st.stop()

st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓", layout="wide")

# --- 2. HEADER ---
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

# --- 3. GESTION DE L'ÉTAT (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# --- 4. CHARGEMENT DES DONNÉES (OPTIMISÉ) ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path):
        return None, "Le dossier 'data' n'existe pas."
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not files:
        return None, "Aucun fichier PDF trouvé."

    try:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
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

# --- 5. MENU SIDEBAR ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    
    if st.button("💬 Chat avec l'IA", use_container_width=True):
        st.session_state.mode = "chat"
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

# --- 6. LOGIQUE PRINCIPALE SELON LE MODE ---

# ==========================================
# MODE 1 : ROADMAP VISUELLE
# ==========================================
if st.session_state.mode == "roadmap":
    st.markdown("### 🗺️ Générateur de Parcours Visuel")
    st.info("Visualisez le cheminement d'une filière sur 5 ans.")
    
    filiere_cible = st.text_input("Quelle filière visualiser ?", placeholder="Ex: Génie Informatique, G. Industriel...")
    
    if st.button("Générer la Roadmap"):
        if filiere_cible:
            with st.spinner("Dessin du graphique..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(f"Programme {filiere_cible} modules années")
                context = "\n".join([d.page_content for d in docs])
                
                graph_prompt = f"""
                Crée un diagramme Graphviz (DOT) pour la filière : {filiere_cible}.
                Contexte: {context}.
                Règles :
                1. Commence par 'digraph G {{ rankdir=LR; node [shape=box, style=filled, fillcolor=lightblue];'.
                2. Nœuds : Année3 -> Année4 -> Année5 -> Métiers.
                3. Dans chaque année, mets 3 modules clés (avec \\n).
                4. Donne UNIQUEMENT le code DOT.
                """
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(graph_prompt)
                dot_code = response.content.replace("```dot", "").replace("```", "").strip()
                try:
                    st.graphviz_chart(dot_code)
                    st.session_state.messages.append({"role": "assistant", "content": f"Roadmap générée pour {filiere_cible}."})
                except:
                    st.error("Erreur graphique.")

# ==========================================
# MODE 2 : COMPARATEUR
# ==========================================
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur Intelligent")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière A", "Génie Informatique")
    f2 = c2.text_input("Filière B", "Génie Industriel")
    
    if st.button("Comparer"):
        with st.spinner("Comparaison..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(f"Infos {f1} et {f2}")
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"""
            Compare {f1} et {f2} sous forme de Tableau Markdown STRICT.
            Critères : Objectif, Modules Clés, Compétences, Débouchés, Salaire.
            Contexte : {context}
            """
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
            st.session_state.messages.append({"role": "assistant", "content": resp.content})

# ==========================================
# MODE 3 : TEST D'ORIENTATION (15 QUESTIONS)
# ==========================================
elif st.session_state.mode == "quiz":
    st.markdown("### 📝 Test d'Orientation Approfondi (15 Questions)")
    st.caption("Prenez le temps de répondre pour une analyse précise de votre profil ingénieur.")

    with st.form("quiz_15"):
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("**🧠 Préférences Générales**")
            q1 = st.radio("1. Qu'aimez-vous le plus ?", ["Concevoir (Théorie)", "Fabriquer (Pratique)", "Organiser (Management)", "Coder (Virtuel)"])
            q2 = st.select_slider("2. Votre niveau en Mathématiques ?", ["Faible", "Moyen", "Bon", "Excellent"])
            q3 = st.radio("3. Environnement de travail ?", ["Bureau / PC", "Terrain / Chantier", "Laboratoire", "Usine / Production"])
            q4 = st.radio("4. Travail en équipe ?", ["Je préfère être autonome", "J'aime collaborer", "Je veux diriger l'équipe"])
            q5 = st.radio("5. Gestion du stress ?", ["Je panique vite", "Je gère bien", "Le stress me motive"])

            st.markdown("**💻 Technique & Info**")
            q6 = st.radio("6. La programmation informatique ?", ["Je déteste", "Ça m'intéresse un peu", "J'adore ça"])
            q7 = st.radio("7. L'Intelligence Artificielle & Big Data ?", ["Pas mon truc", "Curieux", "Je veux en faire mon métier"])
            q8 = st.radio("8. Les réseaux & Télécoms (5G, IoT) ?", ["Bof", "Intéressant", "Passionnant"])

        with col_q2:
            st.markdown("**⚙️ Industriel & Sciences**")
            q9 = st.radio("9. La mécanique et les machines ?", ["Ennuyeux", "Utile", "Fascinant"])
            q10 = st.radio("10. L'électricité et l'électronique ?", ["Trop complexe", "Ça va", "J'aime bricoler/comprendre"])
            q11 = st.radio("11. La logistique (Supply Chain) ?", ["Pas intéress
