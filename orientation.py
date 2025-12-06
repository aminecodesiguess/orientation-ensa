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
    # Fallback si pas de secrets (pour test local rapide)
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

# Nouvelle variable pour savoir si on affiche le QCM ou le Chat
if "show_quiz" not in st.session_state:
    st.session_state.show_quiz = False

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

# --- 5. BARRE LATÉRALE ---
with st.sidebar:
    st.header("🎯 Menu")
    
    # Bouton pour lancer le QCM
    if st.button("📝 Passer le Test d'Orientation"):
        st.session_state.show_quiz = True
    
    # Bouton pour revenir au Chat normal
    if st.button("💬 Retour au Chat"):
        st.session_state.show_quiz = False
        
    st.divider()
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 6. LOGIQUE PRINCIPALE : QCM ou CHAT ? ---

if st.session_state.show_quiz:
    # --- A. MODE QCM (NOUVEAU) ---
    st.markdown("### 📝 Test de Personnalité & Orientation (10 Questions)")
    st.info("Répondez spontanément. L'IA analysera vos réponses pour trouver votre filière idéale.")

    with st.form("quiz_form"):
        # Les 10 Questions Stratégiques
        q1 = st.radio("1. Qu'est-ce qui vous passionne le plus ?", 
                      ["Comprendre comment fonctionnent les choses (Théorie)", "Fabriquer et construire des choses (Pratique)", "Gérer des projets et des équipes", "Le monde du numérique et du code"])
        
        q2 = st.select_slider("2. Aimez-vous les Mathématiques ?", options=["Pas du tout", "Moyen", "J'aime bien", "J'adore"])
        
        q3 = st.radio("3. Quel type d'environnement de travail préférez-vous ?", 
                      ["Bureau calme devant un ordinateur", "Terrain / Chantier / Usine", "Laboratoire de recherche", "Réunions et Management"])
        
        q4 = st.radio("4. Face à un problème, vous êtes plutôt :", 
                      ["Analytique (Je cherche la cause logique)", "Créatif (J'invente une solution nouvelle)", "Pragmatique (Je veux que ça marche vite)", "Organisé (Je planifie la résolution)"])
        
        q5 = st.radio("5. Quel domaine vous attire le moins ?", 
                      ["La Chimie et la Biologie", "L'Informatique", "La Mécanique et l'Électricité", "L'Économie et la Gestion"])
        
        q6 = st.radio("6. Aimez-vous programmer / coder ?", ["Non, ça m'ennuie", "Un peu, par curiosité", "Oui, je pourrais y passer des heures"])
        
        q7 = st.radio("7. L'écologie et l'environnement sont pour vous :", ["Un sujet intéressant", "Une priorité absolue dans mon futur métier", "Secondaire par rapport à la technologie"])
        
        q8 = st.radio("8. Préférez-vous travailler sur :", ["Du logiciel (Virtuel)", "Du matériel (Hardware, Machines, Robots)", "Des processus (Organisation, Logistique)"])
        
        q9 = st.radio("9. Comment gérez-vous le stress ?", ["Je panique un peu", "Je reste calme et concentré", "J'ai besoin d'action"])
        
        q10 = st.text_input("10. En un mot, quel est votre métier de rêve ? (ex: Chef de projet, Data Scientist, Ingénieur BTP...)")

        submitted = st.form_submit_button("🎓 Analyser mes réponses")

        if submitted:
            with st.spinner("L'IA croise vos réponses avec les filières de l'ENSA..."):
                # 1. Récupération contexte
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke("Liste des filières génie informatique industriel civil télécom éco")
                context_filieres = "\n\n".join([doc.page_content for doc in relevant_docs])

                # 2. Construction du Prompt avec les réponses du QCM
                quiz_summary = f"""
                R1 (Passion): {q1}
                R2 (Maths): {q2}
                R3 (Environnement): {q3}
                R4 (Résolution): {q4}
                R5 (Aime moins): {q5}
                R6 (Code): {q6}
                R7 (Écologie): {q7}
                R8 (Support): {q8}
                R9 (Stress): {q9}
                R10 (Rêve): {q10}
                """

                final_prompt = f"""
                Tu es un conseiller d'orientation expert de l'ENSA Tanger.
                
                MISSION : 
                Analyse les réponses de l'étudiant au QCM ci-dessous.
                Déduis son profil psychologique et technique.
                Recommande-lui LA filière la plus adaptée parmi celles disponibles dans le contexte.

                REPONSES DE L'ÉTUDIANT (QCM) :
                {quiz_summary}

                CONTEXTE DES FILIÈRES DISPONIBLES :
                {context_filieres}

                FORMAT DE LA RÉPONSE :
                1. 🧠 **Analyse de Profil** : Tes points forts et intérêts détectés.
                2. 🏆 **Filière Recommandée** : Le nom précis de la filière.
                3. 🚀 **Pourquoi ce choix ?** : Explication détaillée faisant le lien entre le QCM et la filière.
                """

                # 3. Appel IA
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                response = llm.invoke(final_prompt)
                
                # 4. Affichage
                st.success("Analyse terminée !")
                st.markdown(response.content)
                
                # Ajout à l'historique pour qu'on puisse en discuter après
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat du Test QCM :**\n{response.content}"})
                st.balloons() # Petit effet visuel sympa

else:
    # --- B. MODE CHAT (ANCIEN CODE) ---
    # Affichage historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Chat
    if prompt := st.chat_input("Posez une question sur l'école ou sur votre résultat..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("..."):
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
                
                # Prompt système qui prend en compte l'historique récent pour la cohérence
                system_prompt = f"""Tu es un expert de l'ENSA Tanger.
                Utilise le contexte suivant pour répondre.
                Contexte : {context}
                Question : {prompt}
                """
                
                response = llm.invoke(system_prompt)
                st.markdown(response.content)
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})
