import streamlit as st
from pypdf import PdfReader
from datetime import datetime
import gc 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq



@st.cache_resource
def load_embeddings():
    
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  
        model_kwargs={'device': 'cpu'}
    )

@st.cache_data(ttl=3600)  
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        text += "".join(page.extract_text() or "" for page in reader.pages)
    return text

def get_vector_store(raw_text):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(raw_text)
    embeddings = load_embeddings()
    return FAISS.from_texts(chunks[:50], embeddings)  

def get_groq_response(question, vector_store, api_key, model="llama-3.1-8b-instant"):
    docs = vector_store.similarity_search(question, k=2)  # Only 2 docs
    context = "\n".join([doc.page_content for doc in docs])
    
    client = Groq(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQ: {question}\n\nA:"}],
        temperature=0,
        max_tokens=1000  
    ).choices[0].message.content



def main():
    st.set_page_config(page_title="PDF-BOT", page_icon="⚡", layout="wide")
    
    # Minimal CSS
    st.markdown("""
    <style>
    .stApp {background: #0f0f0f;}
    .header {color: #6A5ACD; font-size: 3rem; font-weight: 800;}
    .btn-quick {background: #10B981; color: white; border-radius: 12px;}
    .btn-summary {background: #3B82F6; color: white; border-radius: 12px;}
    </style>
    """, unsafe_allow_html=True)

    # Session State
    for key in ['history', 'vector_store', 'current_q', 'current_a']:
        if key not in st.session_state:
            st.session_state[key] = "" if key in ['current_q', 'current_a'] else []

    # ---------------- HEADER ---------------- #
    st.markdown('<h1 class="header">PDF-BOT</h1>', unsafe_allow_html=True)
    st.markdown("### Upload → Click → Instant Answer")

    # ---------------- QUICK BUTTONS ---------------- #
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Key Points", key="keypoints", help="5-8 bullet points"):
            st.session_state.current_q = "List 5-8 KEY POINTS from this document as bullets:"
    
    with col2:
        if st.button("📄 Summary", key="summary", help="Full document overview"):
            st.session_state.current_q = "Summarize the ENTIRE document in 200-300 words:"

    if st.session_state.current_q:
        st.markdown("### 💬 Answer")
        st.info(st.session_state.current_q)
        if st.session_state.current_a:
            st.success(st.session_state.current_a)
        else:
            st.info("👇 Setup PDFs first")

    
    st.markdown("---")
    q = st.text_input("Or ask anything:", value=st.session_state.current_q)

    
    with st.sidebar:
        st.header("⚙️ Quick Setup")
        
        api_key = st.text_input("Groq Key", type="password")
        pdfs = st.file_uploader("PDFs", accept_multiple_files=True, type="pdf")
        
        if st.button("🚀 Process"):
            if pdfs and api_key:
                with st.spinner("Processing please wait.."):
                    text = get_pdf_text(pdfs)
                    st.session_state.vector_store = get_vector_store(text)
                    st.success("✅ Ready!")
                    gc.collect()  
            else:
                st.error("PDFs + Key needed")

    
        st.markdown("### 📜 History")
        for h in st.session_state.history[-3:]:
            st.caption(f"• {h[:80]}...")

  
    if q and st.session_state.vector_store and api_key:
        if q != st.session_state.current_q or not st.session_state.current_a:
            with st.spinner("Let me think..."):
                answer = get_groq_response(q, st.session_state.vector_store, api_key)
                st.session_state.current_q = q
                st.session_state.current_a = answer
                st.session_state.history.append(f"Q: {q[:50]} | A: {answer[:50]}...")
            st.rerun()


if __name__ == "__main__":
    main()