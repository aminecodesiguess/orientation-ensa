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

# --- 3. GESTION DE L'ÉTAT (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Nouvelle variable pour gérer les 3 modes : "chat", "quiz", "compare"
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

st.session_state.vectorstore = vectorstore

# --- 5. BARRE LATÉRALE (MENU PRINCIPAL) ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    
    # Navigation avec des boutons qui changent l'état "mode"
    if st.button("💬 Chat avec l'IA", use_container_width=True):
        st.session_state.mode = "chat"
        
    if st.button("📝 Test d'Orientation", use_container_width=True):
        st.session_state.mode = "quiz"
        
    if st.button("⚖️ Comparateur de Filières", use_container_width=True):
        st.session_state.mode = "compare"
        
    st.divider()
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# --- 6. LOGIQUE PRINCIPALE SELON LE MODE ---

# ==========================================
# MODE 1 : COMPARATEUR (NOUVEAU 🌟)
# ==========================================
if st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur Intelligent")
    st.info("Entrez deux filières ou concepts pour obtenir un tableau comparatif détaillé.")

    col_a, col_b = st.columns(2)
    with col_a:
        filiere_1 = st.text_input("Filière A", placeholder="Ex: Génie Informatique")
    with col_b:
        filiere_2 = st.text_input("Filière B", placeholder="Ex: Génie Industriel")

    if st.button("Générer le Tableau Comparatif", type="primary"):
        if filiere_1 and filiere_2:
            with st.spinner(f"Comparaison entre {filiere_1} et {filiere_2}..."):
                # 1. Recherche large pour avoir des infos sur les deux sujets
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                query = f"Informations complètes sur {filiere_1} et {filiere_2}, matières, débouchés, salaire"
                relevant_docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # 2. Prompt "Strict Markdown Table"
                compare_prompt = f"""
                Tu es un expert de l'ENSA Tanger.
                
                TA MISSION :
                Comparer objectivement "{filiere_1}" et "{filiere_2}" en te basant sur le contexte fourni.
                
                FORMAT OBLIGATOIRE :
                Tu dois répondre UNIQUEMENT avec un Tableau Markdown.
                Les colonnes doivent être : Critère | {filiere_1} | {filiere_2}
                
                Les critères (lignes) doivent inclure :
                - Objectif de la formation
                - Modules principaux (Matières clés)
                - Compétences développées
                - Types de débouchés (Métiers)
                - Secteurs d'activité
                - Point fort majeur
                
                CONTEXTE :
                {context}
                
                Si une information manque dans le contexte, mets "N/A" ou "Non précisé".
                """

                # 3. Génération
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(compare_prompt)
                
                # 4. Affichage
                st.markdown(response.content)
                
                # On sauvegarde dans l'historique pour garder une trace
                st.session_state.messages.append({"role": "assistant", "content": f"**Comparaison demandée :**\n{response.content}"})

        else:
            st.warning("Veuillez remplir les deux champs.")

# ==========================================
# MODE 2 : QCM (QUIZ)
# ==========================================
elif st.session_state.mode == "quiz":
    st.markdown("### 📝 Test de Personnalité & Orientation")
    st.caption("Répondez spontanément pour découvrir votre profil.")

    with st.form("quiz_form"):
        # Questions (Version compacte pour la lisibilité du code)
        q1 = st.radio("1. Passion ?", ["Théorie & Maths", "Pratique & Fabrication", "Management & Équipe", "Code & Virtuel"])
        q2 = st.select_slider("2. Niveau en Maths ?", ["Faible", "Moyen", "Bon", "Excellent"])
        q3 = st.radio("3. Environnement ?", ["Bureau / PC", "Terrain / Usine", "Labo R&D"])
        q4 = st.radio("4. Approche ?", ["Analytique", "Créative", "Pragmatique"])
        q5 = st.radio("5. A éviter ?", ["Chimie/Bio", "Informatique", "Mécanique/Élec", "Économie"])
        
        # ... Tu peux remettre les 10 questions ici si tu veux, j'ai abrégé pour l'exemple ...
        
        submitted = st.form_submit_button("🎓 Analyser mon profil")

        if submitted:
            with st.spinner("Analyse du profil..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Liste des filières")
                context = "\n".join([d.page_content for d in docs])
                
                summary = f"Passion: {q1}, Maths: {q2}, Env: {q3}, Style: {q4}, Evite: {q5}"
                
                prompt = f"""
                Agis comme un conseiller d'orientation. Analyse ce profil étudiant : {summary}.
                Sur la base de ces documents : {context}.
                Recommande la meilleure filière ENSA Tanger avec une justification précise.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                resp = llm.invoke(prompt)
                st.success("Résultat :")
                st.markdown(resp.content)
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat QCM :**\n{resp.content}"})

# ==========================================
# MODE 3 : CHAT (CLASSIC)
# ==========================================
elif st.session_state.mode == "chat":
    # Affichage historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    if prompt := st.chat_input("Posez votre question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("..."):
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                
                sys_prompt = f"""Expert ENSA Tanger. Contexte: {context}. Question: {prompt}."""
                
                response = llm.invoke(sys_prompt)
                st.markdown(response.content)
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})
