import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- TA CLÉ API ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Clé API non trouvée dans les secrets.")
    st.stop()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓")

# --- HEADER ---
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

# --- INITIALISATION DE LA MÉMOIRE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FONCTION DE CHARGEMENT OPTIMISÉE (CACHÉE) ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    """
    Cette fonction lit les PDF du dossier 'data' une seule fois
    et garde le résultat en mémoire pour tout le monde.
    """
    folder_path = "data" # Le nom de ton dossier
    all_docs = []
    
    # Vérification que le dossier existe
    if not os.path.exists(folder_path):
        return None, "Le dossier 'data' n'existe pas."
    
    # Liste des fichiers PDF
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None, "Aucun fichier PDF trouvé dans le dossier 'data'."

    # Chargement des fichiers
    try:
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
            
        # Découpage
       text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,    # Morceaux plus petits (plus précis)
            chunk_overlap=100, # Moins de chevauchement nécessaire
            separators=["\n\n", "\n", ".", " ", ""] # Coupe de préférence aux paragraphes
        )
        splits = text_splitter.split_documents(all_docs)
        
        # Vectorisation
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore, None
        
    except Exception as e:
        return None, str(e)

# --- CHARGEMENT AUTOMATIQUE AU DÉMARRAGE ---
with st.spinner("Chargement de la base de connaissances de l'école..."):
    vectorstore, error_msg = initialize_vectorstore()

if error_msg:
    st.error(f"Erreur critique : {error_msg}")
    st.stop()

# On stocke le vectorstore dans la session pour l'utiliser plus bas
st.session_state.vectorstore = vectorstore

# --- BARRE LATÉRALE (SIMPLIFIÉE) ---
with st.sidebar:
    st.header("🎯 Orientation")
    st.info("Les documents officiels sont déjà chargés.")
    recommend_btn = st.button("Quelle filière est faite pour moi ?")
    
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- LOGIQUE DE RECOMMANDATION ---
if recommend_btn:
    if len(st.session_state.messages) < 2:
        st.warning("Discutez un peu avec moi d'abord (vos goûts, vos envies) !")
    else:
        with st.spinner("Analyse de votre profil..."):
            try:
                # Récupération contextuelle des filières
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke("Liste des filières, spécialités et objectifs.")
                context_filieres = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Historique
                history_text = ""
                for msg in st.session_state.messages:
                    history_text += f"{msg['role'].upper()}: {msg['content']}\n"

                # Prompt
                final_prompt = f"""
                Tu es un conseiller d'orientation expert de l'ENSA Tanger.
                Analyse la conversation ci-dessous pour comprendre le profil de l'étudiant.
                Recommande-lui LA meilleure filière parmi celles disponibles dans le contexte.
                
                CONTEXTE FILIÈRES : {context_filieres}
                HISTORIQUE : {history_text}
                
                Réponds avec :
                1. 🧠 Analyse du profil
                2. 🏆 Recommandation
                3. 💡 Justification
                """

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(final_prompt)
                
                st.markdown("### 🎯 Résultat")
                st.success(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

            except Exception as e:
                st.error(f"Erreur : {e}")

# --- ZONE DE CHAT ---
# Affichage historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nouvelle question
if prompt := st.chat_input("Bonjour, je voudrais des infos sur les filières..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("..."):
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
            
            system_prompt = f"""Tu es un expert de l'ENSA Tanger.
            Réponds à la question en te basant sur le contexte.
            Contexte : {context}
            Question : {prompt}
            """
            
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
    
    st.session_state.messages.append({"role": "assistant", "content": response.content})

