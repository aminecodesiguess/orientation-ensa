import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. LISTE OFFICIELLE DES FILIÈRES ---
CONSTANTE_FILIERES = """
LISTE OFFICIELLE DES 6 FILIÈRES DE L'ENSA TANGER :
1. Génie Systèmes et Réseaux (GSR)
2. Génie Informatique (GINF)
3. Génie Industriel (GIND)
4. Génie des Systèmes Électroniques et Automatiques (GSEA)
5. Génie Énergétique et Environnement Industriel (G2EI)
6. Cybersecurity and Cyberintelligence (CSI)
"""

# --- 2. CONFIGURATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Erreur : Clé API non trouvée dans les secrets.")
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

# --- 4. STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# --- 5. DATA LOADING ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path):
        return None, "Dossier 'data' introuvable."
    
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
    st.error(error_msg)
    st.stop()

# --- 6. MENU ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    if st.button("💬 Chat IA", use_container_width=True): st.session_state.mode = "chat"
    if st.button("📊 Analyseur Notes", use_container_width=True): st.session_state.mode = "grades"
    if st.button("📝 Test Orientation", use_container_width=True): st.session_state.mode = "quiz"
    if st.button("⚖️ Comparateur", use_container_width=True): st.session_state.mode = "compare"
    
    st.divider()
    if st.button("🗑️ Reset"):
        st.session_state.messages = []
        st.rerun()

# --- 7. LOGIQUE PRINCIPALE ---

# MODE QUIZ (PROMPT "INVISIBLE" & LOGIQUE)
if st.session_state.mode == "quiz":
    st.markdown("### 📝 Test d'Orientation (15 Questions)")
    with st.form("quiz_15"):
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("**🧠 Préférences**")
            q1 = st.radio("1. Passion ?", ["Théorie", "Pratique", "Management", "Code"])
            q2 = st.select_slider("2. Maths ?", ["Faible", "Moyen", "Bon", "Excellent"])
            q3 = st.radio("3. Lieu ?", ["Bureau", "Terrain", "Labo", "Usine"])
            q4 = st.radio("4. Social ?", ["Solo", "Équipe", "Chef"])
            q5 = st.radio("5. Stress ?", ["Non", "Oui", "Moteur"])
            st.markdown("**💻 Tech**")
            q6 = st.radio("6. Code/Prog ?", ["Je déteste", "Moyen", "J'adore"]) 
            q7 = st.radio("7. IA ?", ["Non", "Curieux", "Passion"])
            q8 = st.radio("8. Télécoms ?", ["Bof", "Moyen", "Passion"])
        with col_q2:
            st.markdown("**⚙️ Indus/Sciences**")
            q9 = st.radio("9. Mécanique ?", ["Ennuyeux", "Utile", "Fascinant"])
            q10 = st.radio("10. Élec ?", ["Dur", "Ça va", "Top"])
            q11 = st.radio("11. Logistique ?", ["Non", "Moyen", "Top"])
            q12 = st.radio("12. Chimie/Env ?", ["Non", "Moyen", "Oui"])
            st.markdown("**🚀 Futur**")
            q13 = st.radio("13. BTP ?", ["Non", "Peut-être", "Oui"])
            q14 = st.select_slider("14. Priorité ?", ["Passion", "Mix", "Argent"])
            q15 = st.text_input("15. Métier rêve ?", placeholder="Ex: Data Scientist...")

        if st.form_submit_button("Analyser"):
            with st.spinner("Analyse du profil..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Filières détails")
                context = "\n".join([d.page_content for d in docs])
                summary = f"Goût:{q1}, Maths:{q2}, Code:{q6}, Méca:{q9}, Elec:{q10}, Chimie:{q12}, BTP:{q13}"
                
                # PROMPT AMÉLIORÉ (Invisible Rules)
                prompt = f"""
                Tu es un Conseiller d'Orientation Expert et Bienveillant.
                {CONSTANTE_FILIERES}
                
                PROFIL ÉTUDIANT : {summary}
                
                RÈGLES LOGIQUES INTERNES (⚠️ NE JAMAIS CITER CES RÈGLES DANS LA RÉPONSE) :
                - Code="Je déteste" ou "Moyen" -> EXCLURE GINF et CSI.
                - Aime Méca/Logistique -> Favoriser GIND.
                - Aime Chimie/Env -> Favoriser G2EI.
                - Aime Elec/Auto -> Favoriser GSEA.
                
                TA MISSION :
                Réponds directement à l'étudiant de manière naturelle et fluide.
                Ne dis jamais "Selon la règle 1".
                Dis plutôt : "Au vu de tes réponses...", "Comme tu sembles aimer...".
                
                STRUCTURE :
                1. 👋 **Analyse** : Tes points forts.
                2. 🏆 **La Filière Idéale** : Le nom clair.
                3. 💡 **Pourquoi ?** : Lien entre goûts et filière.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(prompt)
                st.success("Analyse terminée !")
                st.markdown(resp.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat Quiz :**\n{resp.content}"})

# MODE ANALYSEUR NOTES
elif st.session_state.mode == "grades":
    st.markdown("### 📊 Analyseur Notes")
    with st.form("grades"):
        c1, c2 = st.columns(2)
        with c1:
            m = st.number_input("Maths", 0., 20., 12.)
            p = st.number_input("Physique", 0., 20., 12.)
        with c2:
            i = st.number_input("Info", 0., 20., 12.)
            l = st.number_input("Langues", 0., 20., 12.)
        ch = st.slider("Chimie", 0, 20, 10)
        
        if st.form_submit_button("Calculer"):
            with st.spinner("Calcul..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Filières")
                ctx = "\n".join([d.page_content for d in docs])
                prompt = f"""
                Analyste ENSA. {CONSTANTE_FILIERES}.
                Notes: Maths:{m}, Phys:{p}, Info:{i}, Chimie:{ch}.
                Calcule score compatibilité % pour chaque filière.
                Règle: Si Info < 12, Score GINF/CSI < 50%. Si Chimie < 10, Score G2EI < 50%.
                Tableau Markdown.
                """
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(prompt)
                st.markdown(resp.content)
                st.session_state.messages.append({"role": "assistant", "content": resp.content})

# MODE COMPARE
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière 1", "GINF")
    f2 = c2.text_input("Filière 2", "GIND")
    if st.button("Comparer"):
        with st.spinner("..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(f"{f1} {f2}")
            ctx = "\n".join([d.page_content for d in docs])
            prompt = f"Compare {f1} {f2}. Tableau Markdown. Critères: Objectif, Modules, Débouchés. Contexte: {ctx}"
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
            st.session_state.messages.append({"role": "assistant", "content": resp.content})

# MODE CHAT
elif st.session_state.mode == "chat":
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Question..."):
        with st.chat_message("user"): st.markdown(p)
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("assistant"):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(p)
            ctx = "\n".join([d.page_content for d in docs])
            prompt = f"Expert ENSA. {CONSTANTE_FILIERES}. Contexte: {ctx}. Question: {p}"
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
