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

# --- 6. LOGIQUE PRINCIPALE SELON LE MODE ---

# ==========================================
# MODE 1 : ANALYSEUR DE NOTES (NOUVEAU 🌟)
# ==========================================
if st.session_state.mode == "grades":
    st.markdown("### 📊 Analyseur de Notes & Compatibilité")
    st.info("Entrez vos moyennes actuelles. L'IA calculera votre % de réussite probable par filière.")

    with st.form("grade_form"):
        col1, col2 = st.columns(2)
        with col1:
            note_math = st.number_input("Mathématiques (/20)", 0.0, 20.0, 12.0)
            note_phys = st.number_input("Physique / Élec (/20)", 0.0, 20.0, 12.0)
        with col2:
            note_info = st.number_input("Informatique / Algo (/20)", 0.0, 20.0, 12.0)
            note_lang = st.number_input("Français / Anglais (/20)", 0.0, 20.0, 12.0)
            
        note_chimie = st.slider("Aisance en Chimie/Bio (0=Nul, 20=Expert)", 0, 20, 10)
        
        submitted = st.form_submit_button("📈 Calculer mes Compatibilités")

        if submitted:
            with st.spinner("Calcul des scores de compatibilité..."):
                # Récupération contextuelle
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Prérequis filières matières importantes coefficients")
                context = "\n".join([d.page_content for d in docs])
                
                notes_summary = f"""
                Maths: {note_math}/20
                Physique: {note_phys}/20
                Informatique: {note_info}/20
                Langues: {note_lang}/20
                Chimie/Bio: {note_chimie}/20
                """
                
                prompt = f"""
                Tu es un Analyste Académique de l'ENSA Tanger.
                
                MISSION :
                À partir des notes de l'étudiant, calcule un "Score de Compatibilité" (en %) pour les principales filières de l'école.
                
                NOTES ÉTUDIANT :
                {notes_summary}
                
                CONTEXTE ÉCOLE :
                {context}
                
                FORMAT DE RÉPONSE ATTENDU (Tableau Markdown) :
                | Filière | Score Compatibilité | Analyse Rapide |
                |---|---|---|
                | Génie Informatique | 95% | Vos notes en Info et Maths sont idéales. |
                | ... | ... | ... |
                
                Sois réaliste : Si la note de Maths est faible, le score en Info/Indus doit baisser. Si Chimie est faible, le score en Eco-Energie doit baisser.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Analyse des Notes :**\n{response.content}"})


# ==========================================
# MODE 2 : ROADMAP VISUELLE
# ==========================================
elif st.session_state.mode == "roadmap":
    st.markdown("### 🗺️ Générateur de Parcours Visuel")
    filiere_cible = st.text_input("Quelle filière visualiser ?", placeholder="Ex: Génie Informatique")
    
    if st.button("Générer la Roadmap"):
        if filiere_cible:
            with st.spinner("Dessin du graphique..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(f"Programme {filiere_cible}")
                context = "\n".join([d.page_content for d in docs])
                
                graph_prompt = f"""
                Crée un diagramme Graphviz (DOT) pour : {filiere_cible}.
                Contexte: {context}.
                Règles : digraph G {{ rankdir=LR; node [shape=box, style=filled, fillcolor=lightblue];
                Nœuds : Année3 -> Année4 -> Année5 -> Métiers.
                Mets 3 modules par année. Donne UNIQUEMENT le code DOT.
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
# MODE 3 : COMPARATEUR
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
# MODE 4 : TEST D'ORIENTATION 15 QUESTIONS
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
            with st.spinner("Analyse..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Filières détails")
                context = "\n".join([d.page_content for d in docs])
                summary = f"R1:{q1}, R2:{q2}, R3:{q3}, R6:{q6}, R9:{q9}, R13:{q13}, R15:{q15}..."
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(f"Conseiller ENSA. Profil: {summary}. Contexte: {context}. Recommande filière.")
                st.success("Résultat :")
                st.markdown(resp.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat Quiz :**\n{resp.content}"})

# ==========================================
# MODE 5 : CHAT
# ==========================================
elif st.session_state.mode == "chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Posez votre question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("..."):
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(f"Expert ENSA. Contexte: {context}. Question: {prompt}")
                st.markdown(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
