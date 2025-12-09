import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from fpdf import FPDF

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

# --- 3. FONCTION DE GÉNÉRATION PDF ---
def create_pdf(user_profile, ai_response):
    class PDF(FPDF):
        def header(self):
            if os.path.exists("logo.png"):
                self.image("logo.png", 10, 8, 25)
            self.set_font('Arial', 'B', 15)
            self.cell(80)
            self.cell(30, 10, "Rapport d'Orientation - ENSA Tanger", 0, 0, 'C')
            self.ln(30)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    def clean_text(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Contenu du Profil
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("1. Votre Profil Étudiant"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, clean_text(user_profile))
    pdf.ln(5)

    # Contenu de la Recommandation
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean_text("2. Recommandation de l'IA"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, clean_text(ai_response))
    
    # Signature
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, clean_text("Document généré automatiquement par l'Assistant ENSAT."), ln=True)

    return pdf.output(dest='S').encode('latin-1')

# --- 4. HEADER INTERFACE ---
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

# --- 5. STATE (GESTION DE LA MÉMOIRE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = None

# Variables spécifiques pour stocker les résultats et éviter l'écran blanc
if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None
if "grades_result" not in st.session_state:
    st.session_state.grades_result = None

# --- 6. DATA LOADING ---
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

# --- 7. MENU ---
with st.sidebar:
    st.header("🎯 Menu Principal")
    if st.button("💬 Chat IA", use_container_width=True): st.session_state.mode = "chat"
    if st.button("📊 Analyseur Notes", use_container_width=True): st.session_state.mode = "grades"
    if st.button("📝 Test Orientation", use_container_width=True): st.session_state.mode = "quiz"
    if st.button("⚖️ Comparateur", use_container_width=True): st.session_state.mode = "compare"
    
    st.divider()
    if st.button("🗑️ Reset"):
        st.session_state.messages = []
        st.session_state.quiz_result = None
        st.session_state.grades_result = None
        st.session_state.last_pdf = None
        st.rerun()

# --- 8. LOGIQUE PRINCIPALE ---

# MODE QUIZ
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
            with st.spinner("Analyse croisée de tes 15 réponses..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Détails modules débouchés filières")
                context = "\n".join([d.page_content for d in docs])
                
                summary = f"""
                PROFIL: Passion={q1}, Maths={q2}, Code={q6}, IA={q7}, 
                Lieu={q3}, Social={q4}, Stress={q5}, Télécoms={q8},
                Méca={q9}, Élec={q10}, Logistique={q11}, Chimie={q12}, BTP={q13},
                Priorité={q14}, Rêve={q15}
                """
                
                prompt = f"""
                Tu es un Expert en Orientation Stratégique à l'ENSA Tanger.
                TES OUTILS : {CONSTANTE_FILIERES}
                DONNÉES CANDIDAT : {summary}
                
                TA MISSION (Analyse Algorithmique) :
                1. LOGIQUE D'ÉLIMINATION :
                   - Code="Je déteste" -> INTERDIRE GINF/CSI.
                   - Maths="Faible" -> ÉVITER GINF/CSI/GSEA.
                   - Chimie="Non" -> ÉVITER G2EI.
                
                2. MATCHING :
                   - GINF : Code="J'adore" + Maths fort.
                   - GIND : Logistique/Méca + Gestion.
                   - GSEA : Élec + Auto.
                   - GSR : Télécoms + Réseaux.
                   - G2EI : Chimie/Env + Énergie.
                   - CSI : IA + Code + Cyber.
                
                FORMAT REPONSE (Markdown) :
                ## 🏆 Ta Filière Idéale : [Nom]
                **Pourquoi :** Explication liée aux réponses.
                ## 🥈 Alternative : [Nom]
                Plan B crédible.
                ## ⚠️ Vigilance
                Point faible à surveiller.
                ## 🔮 Projection
                Métier lié à "{q15}".
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.4)
                resp = llm.invoke(prompt)
                
                # Sauvegarde stable
                st.session_state.quiz_result = resp.content
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat Quiz :**\n\n{resp.content}"})
                
                pdf_bytes = create_pdf(f"Réponses clés Quiz: {summary}", resp.content)
                st.session_state.last_pdf = pdf_bytes
                
                st.rerun()

    # Affichage résultat Quiz (Stable)
    if st.session_state.quiz_result:
        st.divider()
        st.success("✅ Analyse terminée avec succès !")
        st.markdown(st.session_state.quiz_result)
        if st.session_state.last_pdf:
            st.download_button("📄 Télécharger mon Rapport (PDF)", st.session_state.last_pdf, "rapport_orientation_ensa.pdf", "application/pdf")

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
            with st.spinner("Analyse approfondie de tes résultats..."):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke("Prérequis filières matières")
                ctx = "\n".join([d.page_content for d in docs])
                
                summary = f"Mathématiques: {m}/20, Physique: {p}/20, Informatique: {i}/20, Langues: {l}/20, Chimie: {ch}/20"
                
                prompt = f"""
                Tu es le Directeur Pédagogique de l'ENSA Tanger. Tu analyses le dossier d'un étudiant pour l'orienter.
                DONNÉES ÉTUDIANT : {summary}
                CONTEXTE FILIÈRES : {CONSTANTE_FILIERES}
                
                TA MISSION :
                1. Calcule un "Score d'Affinité" (0-100%) pour chaque filière en suivant cette PONDÉRATION LOGIQUE :
                   - GINF & CSI : Coefficient double sur (Maths + Info). Si Info < 12, pénalité forte.
                   - GSEA & G2EI : Coefficient double sur (Physique + Maths).
                   - GIND : Moyenne équilibrée, bonus si Maths & Langues sont solides.
                   - GSR : Mix équilibré Info + Réseaux (considère Info et Maths).
                
                2. Génère un tableau Markdown strict avec les colonnes :
                   | Filière | Score % | Verdict | Conseil Rapide |
                
                3. Ajoute une courte analyse textuelle (3 phrases max) sous le tableau pour résumer ses forces et faiblesses.
                Sois strict mais encourageant.
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.3)
                resp = llm.invoke(prompt)
                
                # Sauvegarde stable pour les notes
                st.session_state.grades_result = resp.content
                st.session_state.messages.append({"role": "assistant", "content": resp.content})
                
                pdf_bytes = create_pdf(f"Relevé de notes: {summary}", resp.content)
                st.session_state.last_pdf = pdf_bytes
                st.rerun()

    # Affichage résultat Notes (Stable)
    if st.session_state.grades_result:
        st.divider()
        st.success("✅ Analyse des notes terminée !")
        st.markdown(st.session_state.grades_result)
        if st.session_state.last_pdf:
            st.download_button("📄 Télécharger mon Bilan (PDF)", st.session_state.last_pdf, "bilan_notes_ensa.pdf", "application/pdf")

# MODE COMPARE
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière 1", "GINF")
    f2 = c2.text_input("Filière 2", "GIND")
    if st.button("Comparer"):
        with st.spinner("Comparaison en cours..."):
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
