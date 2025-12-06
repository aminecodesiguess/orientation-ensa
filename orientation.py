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
# Si tu es en local sans secrets.toml, tu peux remettre ta clé en dur ici temporairement pour tester
# GROQ_API_KEY = "gsk_..." 
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Clé API non trouvée dans les secrets. Veuillez configurer .streamlit/secrets.toml")
    st.stop()

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

# --- INITIALISATION DE LA MÉMOIRE ---
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
    
    st.divider()
    
    # --- NOUVEAU BOUTON DE RECOMMANDATION ---
    st.header("🎯 Orientation")
    recommend_btn = st.button("Quelle filière est faite pour moi ?")

# --- LOGIQUE D'ANALYSE ---
if process_btn:
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
            
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success("✅ Analyse terminée ! Pose ta première question.")
        
        except Exception as e:
            st.error(f"Erreur technique : {e}")

# --- LOGIQUE DE RECOMMANDATION (NOUVEAU) ---
if recommend_btn:
    # On vérifie qu'on a de la matière pour travailler
    if "vectorstore" not in st.session_state:
        st.error("Veuillez d'abord analyser les documents.")
    elif len(st.session_state.messages) < 2:
        st.warning("La discussion est trop courte ! Posez quelques questions d'abord pour que je puisse cerner votre profil.")
    else:
        with st.spinner("Analyse de votre profil et des filières..."):
            try:
                # 1. On récupère le contexte des filières (RAG)
                # On demande explicitement au moteur de chercher la liste des filières
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke("Liste des filières, départements, spécialités et objectifs de formation.")
                context_filieres = "\n\n".join([doc.page_content for doc in relevant_docs])

                # 2. On formate l'historique de la conversation
                history_text = ""
                for msg in st.session_state.messages:
                    history_text += f"{msg['role'].upper()}: {msg['content']}\n"

                # 3. Prompt Spécial "Conseiller"
                final_prompt = f"""
                Tu es un conseiller d'orientation expert de l'ENSA Tanger.
                
                TA MISSION :
                Analyser la conversation ci-dessous pour comprendre le profil de l'étudiant (intérêts, points forts, personnalité).
                Ensuite, recommande-lui LA meilleure filière parmi celles disponibles dans le contexte documentaire.
                
                CONTEXTE DES FILIÈRES DISPONIBLES (Documents) :
                {context_filieres}
                
                HISTORIQUE DE LA CONVERSATION :
                {history_text}
                
                FORMAT DE RÉPONSE ATTENDU :
                1. 🧠 **Analyse de ton profil** : (Résumé de ce que tu as compris de l'étudiant)
                2. 🏆 **Recommandation** : (Nom de la filière recommandée)
                3. 💡 **Pourquoi ?** : (Justification précise liant le profil aux débouchés de la filière)
                """

                # 4. Appel à l'IA
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(final_prompt)
                
                # 5. Affichage du résultat
                st.markdown("### 🎯 Résultat de ton orientation")
                st.success(response.content)
                
                # On ajoute aussi ce résultat à l'historique pour qu'il reste affiché
                st.session_state.messages.append({"role": "assistant", "content": response.content})

            except Exception as e:
                st.error(f"Erreur lors de la recommandation : {e}")


# --- ZONE DE CHAT ---
if "vectorstore" in st.session_state:
    
    # Affichage historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nouvelle question
    if prompt := st.chat_input("Ex: Quelles sont les filières disponibles ?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                
                system_prompt = f"""Tu es un expert de l'ENSA Tanger.
                Utilise le contexte suivant pour répondre à la question.
                
                Contexte : {context}
                Question : {prompt}
                """
                
                response = llm.invoke(system_prompt)
                response_text = response.content
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

elif not uploaded_files:
    st.info("👈 Commencez par charger vos documents PDF dans le menu à gauche.")
