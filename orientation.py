import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. LISTE OFFICIELLE DES FILIÈRES (VÉRITÉ ABSOLUE) ---
CONSTANTE_FILIERES = """
LISTE OFFICIELLE ET EXCLUSIVE DES 6 FILIÈRES DE L'ENSA TANGER (Cycle Ingénieur) :
1. Génie Systèmes et Réseaux (GSR)
2. Génie Informatique (GINF)
3. Génie Industriel (GIND)
4. Génie des Systèmes Électroniques et Automatiques (GSEA)
5. Génie Énergétique et Environnement Industriel (G2EI)
6. Cybersecurity and Cyberintelligence (CSI)

N'invente aucune autre filière.
"""

# --- 2. CONFIGURATION & SÉCURITÉ ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Erreur : Clé API non trouvée dans les secrets (st.secrets).")
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

# --- 5. CHARGEMENT DES DONNÉES (OPTIMISÉ PDF + TXT) ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path):
        return None, "Dossier 'data' introuvable."
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf') or f.endswith('.txt')]
    if not files:
        return None, "Aucun fichier trouvé dans 'data'."

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
    st.error(f"Erreur critique : {error_msg}")
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
# MODE 1 : ANALYSEUR DE NOTES (NOUVEAU 🌟)
# ==========================================
if st.session_state.mode == "grades":
    st.markdown("### 📊 Analyseur de Notes & Compatibilité")
    st.info("Entrez vos moyennes pour voir quelles filières correspondent le mieux à votre profil académique.")

    with st.form("grade_form"):
        c1, c2 = st.columns(2)
        with c1:
            note_math = st.number_input("Mathématiques", 0.0, 20.0, 12.0)
            note_phys = st.number_input("Physique / Élec", 0.0, 20.0, 12.0)
        with c2:
            note_info = st.number_input("Informatique / Algo", 0.0, 20.0, 12.0)
            note_lang = st.number_input("Français / Anglais", 0.0, 20.0, 12.0)
            
        note_chimie = st.slider("Aisance en Chimie/Bio (0=Faible, 20=Fort)", 0, 20, 10)
        
        if st.form_submit_button("📈 Calculer mes Compatibilités"):
            with st.spinner("Calcul scientifique des scores..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Prérequis matières coefficients filières")
                context = "\n".join([d.page_content for d in docs])
                
                notes_summary = f"Maths:{note_math}, Phys:{note_phys}, Info:{note_info}, Lang:{note_lang}, Chimie:{note_chimie}"
                
                prompt = f"""
                Tu es un Analyste Académique ENSA.
                {CONSTANTE_FILIERES}
                
                MISSION : Pour CHAQUE filière de la liste officielle, calcule un score de compatibilité (0-100%) basé sur les notes de l'étudiant.
                
                NOTES ÉTUDIANT : {notes_summary}
                CONTEXTE PDF : {context}
                
                Règles de calcul mental :
                - GINF/GSR/CSI/GSEA demandent beaucoup de Maths et Info.
                - GIND demande Maths + Statistiques.
                - G2EI demande Physique + Thermodynamique (Chimie utile).
                
                Format de réponse : Tableau Markdown (Filière | Score | Analyse courte).
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
    filiere_cible = st.text_input("Quelle filière visualiser ?", placeholder="Ex: GINF, CSI, GSEA...")
    
    if st.button("Générer la Roadmap"):
        if filiere_cible:
            with st.spinner("Dessin du graphique..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(f"Programme {filiere_cible} modules")
                context = "\n".join([d.page_content for d in docs])
                
                graph_prompt = f"""
                Crée un diagramme Graphviz (DOT) pour : {filiere_cible}.
                Contexte: {context}.
                {CONSTANTE_FILIERES}
                Règles : digraph G {{ rankdir=LR; node [shape=box, style=filled, fillcolor=lightblue];
                Nœuds : Année3 -> Année4 -> Année5 -> Métiers.
                Donne UNIQUEMENT le code DOT. Pas de ```.
                """
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(graph_prompt)
                dot_code = response.content.replace("```dot", "").replace("```", "").strip()
                try:
                    st.graphviz_chart(dot_code)
                    st.session_state.messages.append({"role": "assistant", "content": f"Roadmap générée pour {filiere_cible}."})
                except:
                    st.error("Erreur graphique. L'IA a mal formaté le code DOT.")

# ==========================================
# MODE 3 : COMPARATEUR
# ==========================================
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur Intelligent")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière A", "GINF")
    f2 = c2.text_input("Filière B", "GSEA")
    
    if st.button("Comparer"):
        with st.spinner("Comparaison..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(f"Infos {f1} et {f2}")
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"""
            Compare {f1} et {f2} (Tableau Markdown).
            {CONSTANTE_FILIERES}
            Critères : Objectif, Modules Clés, Débouchés, Salaire.
            Contexte : {context}
            """
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
            st.session_state.messages.append({"role": "assistant", "content": resp.content})

# ==========================================
# MODE 4 : TEST D'ORIENTATION (15 Q)
# ==========================================
elif st.session_state.mode == "quiz":
    st.markdown("### 📝 Test d'Orientation (15 Questions)")
    
    with st.form("quiz_15"):
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("**🧠 Préférences**")
            q1 = st.radio("1. Passion ?", ["Théorie", "Pratique", "Management", "Code"])
            q2 = st.select_slider("2. Niveau Maths ?", ["Faible", "Moyen", "Bon", "Excellent"])
            q3 = st.radio("3. Environnement ?", ["Bureau", "Terrain", "Labo", "Usine"])
            q4 = st.radio("4. Équipe ?", ["Autonome", "Collaboratif", "Directeur"])
            q5 = st.radio("5. Stress ?", ["Panique", "Gère bien", "Moteur"])
            st.markdown("**💻 Tech**")
            q6 = st.radio("6. Code ?", ["Déteste", "Moyen", "J'adore"])
            q7 = st.radio("7. IA ?", ["Non", "Curieux", "Passion"])
            q8 = st.radio("8. Télécoms ?", ["Bof", "Intéressant", "Passion"])
        with col_q2:
            st.markdown("**⚙️ Indus & Sciences**")
            q9 = st.radio("9. Mécanique ?", ["Ennuyeux", "Utile", "Fascinant"])
            q10 = st.radio("10. Élec ?", ["Complexe", "Ça va", "Bricoleur"])
            q11 = st.radio("11. Logistique ?", ["Non", "Pourquoi pas", "Stratégique"])
            q12 = st.radio("12. Chimie ?", ["Je fuis", "Neutre", "Avenir"])
            st.markdown("**🚀 Avenir**")
            q13 = st.radio("13. BTP ?", ["Non", "Peut-être", "Oui"])
            q14 = st.select_slider("14. Salaire vs Passion ?", ["Passion", "Équilibré", "Salaire"])
            q15 = st.text_input("15. Rêve ?", placeholder="Ex: Data Scientist...")

        if st.form_submit_button("Analyser"):
            with st.spinner("Analyse approfondie..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Filières détails")
                context = "\n".join([d.page_content for d in docs])
                summary = f"R1:{q1}, R2:{q2}, R6:{q6}, R9:{q9}, R13:{q13}, R15:{q15}"
                
                prompt = f"""
                Conseiller ENSA. Profil : {summary}.
                {CONSTANTE_FILIERES}
                MISSION : Recommande la meilleure filière parmi les 6 officielles.
                Justifie ton choix avec les réponses du quiz.
                Contexte : {context}
                """
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(prompt)
                st.success("Résultat :")
                st.markdown(resp.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat Quiz :**\n{resp.content}"})

# ==========================================
# MODE 5 : CHAT (DÉFAUT)
# ==========================================
elif st.session_state.mode == "chat":
    # Affichage
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Input
    if prompt := st.chat_input("Posez votre question sur l'ENSA..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                
                sys_prompt = f"""
                Tu es un Expert ENSA Tanger.
                {CONSTANTE_FILIERES}
                Contexte PDF : {context}
                Question : {prompt}
                """
                resp = llm.invoke(sys_prompt)
                st.markdown(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
