import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Clé API non trouvée dans les secrets.")
    st.stop()

st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓")

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

# --- 3. STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"

# --- 4. DATA ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path):
        return None, "Dossier 'data' introuvable."
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not files:
        return None, "Aucun PDF trouvé."

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

with st.spinner("Chargement de la base..."):
    vectorstore, error_msg = initialize_vectorstore()

if error_msg:
    st.error(error_msg)
    st.stop()

# --- 5. MENU SIDEBAR ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    
    if st.button("💬 Chat IA", use_container_width=True):
        st.session_state.mode = "chat"
    if st.button("📝 Test Orientation", use_container_width=True):
        st.session_state.mode = "quiz"
    if st.button("⚖️ Comparateur", use_container_width=True):
        st.session_state.mode = "compare"
    # NOUVEAU BOUTON
    if st.button("🗺️ Roadmap Visuelle", use_container_width=True):
        st.session_state.mode = "roadmap"
        
    st.divider()
    if st.button("🗑️ Reset"):
        st.session_state.messages = []
        st.rerun()

# --- 6. MODES ---

# MODE ROADMAP (NOUVEAU 🌟)
if st.session_state.mode == "roadmap":
    st.markdown("### 🗺️ Générateur de Parcours Visuel")
    st.info("Visualisez votre avenir : de la 1ère année jusqu'au métier de vos rêves.")
    
    filiere_cible = st.text_input("Quelle filière voulez-vous visualiser ?", placeholder="Ex: Génie Informatique, G. Industriel...")
    
    if st.button("Générer le Graphique"):
        if filiere_cible:
            with st.spinner("Création du diagramme en cours..."):
                # RAG
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(f"Programme détaillé {filiere_cible} modules années débouchés")
                context = "\n".join([d.page_content for d in docs])
                
                # Prompt pour Graphviz
                graph_prompt = f"""
                Tu es un expert en visualisation de données.
                Crée un diagramme en langage DOT (Graphviz) pour représenter le parcours de la filière : {filiere_cible}.
                
                Utilise le contexte : {context}
                
                Règles strictes :
                1. Le code doit commencer par 'digraph G {{' et finir par '}}'.
                2. Utilise 'rankdir=LR;' (de gauche à droite).
                3. Nœuds : Année 3 (Début cycle ingénieur), Année 4, Année 5, et 3 Métiers de sortie.
                4. Relie les années entre elles et l'Année 5 aux métiers.
                5. Dans chaque boîte (nœud) Année, liste 2 ou 3 modules clés (avec \\n pour saut de ligne).
                6. Donne UNIQUEMENT le code DOT, rien d'autre. Pas de ``` au début ni à la fin.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(graph_prompt)
                
                # Nettoyage du code (par sécurité)
                dot_code = response.content.replace("```dot", "").replace("```", "").strip()
                
                try:
                    st.graphviz_chart(dot_code)
                    st.success(f"Voici le parcours type pour {filiere_cible}")
                    st.session_state.messages.append({"role": "assistant", "content": f"J'ai généré la roadmap pour {filiere_cible}."})
                except:
                    st.error("Erreur de génération du graphique. Réessayez.")
                    st.code(dot_code) # Debug
        else:
            st.warning("Entrez un nom de filière.")

# MODE COMPARATEUR
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur Intelligent")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière 1", "Génie Informatique")
    f2 = c2.text_input("Filière 2", "Génie Industriel")
    
    if st.button("Comparer"):
        with st.spinner("Comparaison..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(f"Infos {f1} et {f2}")
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"""
            Compare {f1} et {f2} sous forme de Tableau Markdown.
            Critères : Objectif, Modules Clés, Débouchés, Salaire estimé.
            Contexte : {context}
            """
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
            st.session_state.messages.append({"role": "assistant", "content": resp.content})

# MODE QUIZ
elif st.session_state.mode == "quiz":
    st.markdown("### 📝 Test Orientation")
    with st.form("quiz"):
        q1 = st.radio("Domaine préféré ?", ["Informatique", "Industrie", "BTP", "Télécom"])
        q2 = st.select_slider("Aisance Mathématique", ["Faible", "Moyenne", "Forte"])
        # ... ajoute tes autres questions ici ...
        if st.form_submit_button("Analyser"):
            with st.spinner("Analyse..."):
                # ... (Logique identique à avant) ...
                # Pour l'exemple je simplifie l'appel
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                res = llm.invoke(f"Conseille une filière pour qqn qui aime {q1} avec niveau maths {q2}")
                st.success("Résultat :")
                st.markdown(res.content)
                st.session_state.messages.append({"role": "assistant", "content": res.content})

# MODE CHAT
elif st.session_state.mode == "chat":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(prompt)
            ctx = "\n".join([d.page_content for d in docs])
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            resp = llm.invoke(f"Expert ENSA. Contexte: {ctx}. Question: {prompt}")
            st.markdown(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
