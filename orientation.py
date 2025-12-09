import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from fpdf import FPDF

# --- 1. CONFIGURATION & SÉCURITÉ ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("Erreur critique : Clé API introuvable dans les secrets.")
    st.stop()

st.set_page_config(page_title="Orientation ENSA Tanger", page_icon="🎓", layout="wide")

# --- 2. MODÈLE SÉCURISÉ (ANTI-PLANTAGE) ---
# On utilise la version 8b car elle est très rapide et ne bloque jamais en démo.
MODEL_NAME = "llama3-8b-8192"

# --- 3. LISTE OFFICIELLE ---
CONSTANTE_FILIERES = """
LISTE OFFICIELLE DES 6 FILIÈRES DE L'ENSA TANGER :
1. Génie Systèmes et Réseaux (GSR)
2. Génie Informatique (GINF)
3. Génie Industriel (GIND)
4. Génie des Systèmes Électroniques et Automatiques (GSEA)
5. Génie Énergétique et Environnement Industriel (G2EI)
6. Cybersecurity and Cyberintelligence (CSI)
"""

# --- 4. FONCTION PDF ---
def create_pdf(user_profile, ai_response):
    class PDF(FPDF):
        def header(self):
            if os.path.exists("logo.png"):
                self.image("logo.png", 10, 8, 25)
            self.set_font('Arial', 'B', 15)
            self.cell(80)
            self.cell(30, 10, "Rapport Orientation ENSAT", 0, 0, 'C')
            self.ln(30)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    def clean(text):
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean("1. Profil Etudiant"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, clean(user_profile))
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, clean("2. Recommandation IA"), ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, clean(ai_response))
    
    return pdf.output(dest='S').encode('latin-1')

# --- 5. CHARGEMENT DONNÉES ---
@st.cache_resource(show_spinner=False)
def initialize_vectorstore():
    folder_path = "data"
    all_docs = []
    
    if not os.path.exists(folder_path): return None, "Dossier 'data' introuvable."
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf') or f.endswith('.txt')]
    if not files: return None, "Aucun fichier trouvé."

    try:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.pdf'): loader = PyPDFLoader(file_path)
            else: loader = TextLoader(file_path, encoding='utf-8')
            all_docs.extend(loader.load())
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(splits, embeddings), None
    except Exception as e:
        return None, str(e)

# --- 6. INTERFACE ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists("logo.png"): st.image("logo.png", width=100)
    else: st.markdown("# 🏫")
with col2:
    st.title("Assistant Orientation ENSAT")
    st.markdown("**National School of Applied Sciences of Tangier**")
st.divider()

if "messages" not in st.session_state: st.session_state.messages = []
if "mode" not in st.session_state: st.session_state.mode = "chat"
if "last_pdf" not in st.session_state: st.session_state.last_pdf = None

with st.spinner("Chargement..."):
    vectorstore, err = initialize_vectorstore()
    if err: st.error(err)

# --- 7. MENU ---
with st.sidebar:
    st.header("🎯 Menu")
    if st.button("💬 Chat IA", use_container_width=True): st.session_state.mode = "chat"
    if st.button("📊 Analyseur Notes", use_container_width=True): st.session_state.mode = "grades"
    if st.button("📝 Test (15 Q)", use_container_width=True): st.session_state.mode = "quiz"
    if st.button("⚖️ Comparateur", use_container_width=True): st.session_state.mode = "compare"
    st.divider()
    if st.button("🗑️ Reset"):
        st.session_state.messages = []
        st.session_state.last_pdf = None
        st.rerun()

# --- 8. LOGIQUE PRINCIPALE ---

# MODE QUIZ
if st.session_state.mode == "quiz":
    st.markdown("### 📝 Test d'Orientation (15 Questions)")
    with st.form("quiz_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧠 Profil**")
            q1 = st.radio("1. Passion ?", ["Théorie", "Pratique", "Management", "Code"])
            q2 = st.select_slider("2. Maths ?", ["Faible", "Moyen", "Bon", "Excellent"])
            q3 = st.radio("3. Lieu ?", ["Bureau", "Terrain", "Labo", "Usine"])
            q4 = st.radio("4. Social ?", ["Solo", "Équipe", "Chef"])
            q5 = st.radio("5. Stress ?", ["Non", "Oui", "Moteur"])
            st.markdown("**💻 Tech**")
            q6 = st.radio("6. Code ?", ["Je déteste", "Moyen", "J'adore"])
            q7 = st.radio("7. IA ?", ["Non", "Curieux", "Passion"])
            q8 = st.radio("8. Télécoms ?", ["Bof", "Moyen", "Passion"])
        with c2:
            st.markdown("**⚙️ Indus**")
            q9 = st.radio("9. Mécanique ?", ["Ennuyeux", "Utile", "Fascinant"])
            q10 = st.radio("10. Élec ?", ["Dur", "Ça va", "Top"])
            q11 = st.radio("11. Logistique ?", ["Non", "Moyen", "Top"])
            q12 = st.radio("12. Chimie ?", ["Non", "Moyen", "Oui"])
            st.markdown("**🚀 Avenir**")
            q13 = st.radio("13. BTP ?", ["Non", "Peut-être", "Oui"])
            q14 = st.select_slider("14. Priorité ?", ["Passion", "Mix", "Argent"])
            q15 = st.text_input("15. Métier rêve ?", placeholder="Ex: Data Scientist")

        if st.form_submit_button("🎓 Analyser"):
            with st.spinner("Analyse..."):
                retriever = vectorstore.as_retriever()
                ctx = "\n".join([d.page_content for d in retriever.invoke("Filières")])
                summ = f"Goût:{q1}, Maths:{q2}, Code:{q6}, Méca:{q9}, Elec:{q10}, Chimie:{q12}, BTP:{q13}"
                
                prompt = f"""
                Conseiller ENSA. {CONSTANTE_FILIERES}.
                Profil: {summ}.
                Règles STRICTES:
                - Code="Je déteste"/"Moyen" -> EXCLURE GINF/CSI.
                - Aime Méca -> GIND.
                - Aime Chimie -> G2EI.
                - Aime Elec -> GSEA.
                Réponds naturellement. Contexte: {ctx}
                """
                
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
                resp = llm.invoke(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": f"**Résultat Quiz :**\n{resp.content}"})
                st.session_state.last_pdf = create_pdf(f"Réponses: {summ}", resp.content)
                st.rerun()

    if st.session_state.messages and "Résultat" in str(st.session_state.messages[-1].get("content", "")):
        st.success("Terminé !")
        st.markdown(st.session_state.messages[-1]["content"])
        if st.session_state.last_pdf:
            st.download_button("📄 Télécharger Rapport PDF", st.session_state.last_pdf, "rapport.pdf", "application/pdf")

# MODE NOTES
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
                ctx = "\n".join([d.page_content for d in retriever.invoke("Filières")])
                summ = f"M:{m}, P:{p}, I:{i}, Ch:{ch}"
                
                prompt = f"Analyste ENSA. {CONSTANTE_FILIERES}. Notes: {summ}. Tableau Markdown compatibilité %."
                llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
                resp = llm.invoke(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": resp.content})
                st.session_state.last_pdf = create_pdf(f"Notes: {summ}", resp.content)
                st.rerun()

    if st.session_state.messages and ("Tableau" in str(st.session_state.messages[-1].get("content", "")) or "Analyse" in str(st.session_state.messages[-1].get("content", ""))):
        st.markdown(st.session_state.messages[-1]["content"])
        if st.session_state.last_pdf:
            st.download_button("📄 Télécharger Bilan PDF", st.session_state.last_pdf, "bilan.pdf", "application/pdf")

# MODE COMPARE
elif st.session_state.mode == "compare":
    st.markdown("### ⚖️ Comparateur")
    c1, c2 = st.columns(2)
    f1 = c1.text_input("Filière A", "GINF")
    f2 = c2.text_input("Filière B", "GIND")
    if st.button("Comparer"):
        with st.spinner("..."):
            retriever = vectorstore.as_retriever()
            ctx = "\n".join([d.page_content for d in retriever.invoke(f"{f1} {f2}")])
            prompt = f"Compare {f1} et {f2}. Tableau Markdown. Critères: Objectif, Modules, Débouchés. Contexte: {ctx}"
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
            st.session_state.messages.append({"role": "assistant", "content": resp.content})

# MODE CHAT
elif st.session_state.mode == "chat":
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Question sur l'école..."):
        with st.chat_message("user"): st.markdown(p)
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("assistant"):
            retriever = vectorstore.as_retriever()
            ctx = "\n".join([d.page_content for d in retriever.invoke(p)])
            prompt = f"Expert ENSA. {CONSTANTE_FILIERES}. Contexte: {ctx}. Question: {p}"
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
            resp = llm.invoke(prompt)
            st.markdown(resp.content)
        st.session_state.messages.append({"role": "assistant", "content": resp.content})
