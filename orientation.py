import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- TA CLÉ API ---
# Connexion au coffre-fort Streamlit
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓")

# --- HEADER (Gestion d'erreur de l'image) ---
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

# --- INITIALISATION DE LA MÉMOIRE (NOUVEAU) ---
# Si l'historique n'existe pas encore, on crée une liste vide
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("📚 Documents")
    uploaded_files = st.file_uploader(
        "Chargez les PDF ici",
        type="pdf",
        accept_multiple_files=True
    )
    process_btn = st.button("Analyser les documents")

# --- LOGIQUE D'ANALYSE (Inchangée) ---
if process_btn:
    if not GROQ_API_KEY.startswith("gsk_"):
        st.error("⚠️ ATTENTION : Tu n'as pas remplacé la clé API dans le code !")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️ Veuillez d'abord uploader des fichiers PDF.")
        st.stop()

    with st.spinner("Analyse en cours..."):
        try:
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # On stocke la base de données vectorielle
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success("✅ Analyse terminée ! Pose ta première question.")

        except Exception as e:
            st.error(f"Erreur technique : {e}")

# --- ZONE DE CHAT (AMÉLIORÉE) ---

# 1. On vérifie si la base vectorielle est prête
if "vectorstore" in st.session_state:

    # 2. On affiche TOUS les messages précédents de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. On capture la NOUVELLE question
    if prompt := st.chat_input("Ex: Quelles sont les filières disponibles ?"):
        # A. On affiche la question de l'utilisateur tout de suite
        with st.chat_message("user"):
            st.markdown(prompt)
        # B. On l'ajoute à la mémoire
        st.session_state.messages.append({"role": "user", "content": prompt})

        # C. Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                # Récupération du contexte
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Appel à l'IA
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

                # Prompt système
                system_prompt = f"""Tu es un expert de l'ENSA Tanger.
                Utilise le contexte suivant pour répondre à la question.
                Si tu ne sais pas, dis-le.

                Contexte : {context}
                Question : {prompt}
                """

                response = llm.invoke(system_prompt)
                response_text = response.content

                # Affichage de la réponse
                st.markdown(response_text)

        # D. On ajoute la réponse de l'IA à la mémoire
        st.session_state.messages.append({"role": "assistant", "content": response_text})

elif not uploaded_files:
    # Message d'accueil si rien n'est chargé

    st.info("👈 Commencez par charger vos documents PDF dans le menu à gauche.")
