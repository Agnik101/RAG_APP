import streamlit as st
from pypdf import PdfReader
from datetime import datetime
import gc 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_text(raw_text)
    embeddings = load_embeddings()
    
    max_chunks = min(len(chunks), 200)
    return FAISS.from_texts(chunks[:max_chunks], embeddings)

def get_groq_response(question, vector_store, api_key, model="llama-3.1-8b-instant"):
    docs = vector_store.similarity_search(question, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    
    client = Groq(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "Not found in document".

CONTEXT:
{context}

QUESTION:
{question}

Give a clear, structured answer.
"""}],
        temperature=0,
        max_tokens=1000  
    ).choices[0].message.content

def main():
    st.set_page_config(page_title="PDF-BOT", page_icon="⚡", layout="wide")
    
    # 🔥 PURE RED BUTTONS - ALL IDENTICAL
    st.markdown("""
    <style>
    .stApp {background: #0a0a0a;}
    .header {color: #EF4444; font-size: 3rem; font-weight: 800; text-shadow: 0 0 20px rgba(239,68,68,0.5);}
    h1, h2, h3, h4, h5, h6 {color: #EF4444 !important;}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #EF4444 !important;}
    
    /* ALL BUTTONS - PURE RED */
    .stButton > button {
        background: #EF4444 !important;
        color: white !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 15px rgba(239,68,68,0.5) !important;
        border: 2px solid #DC2626 !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: #DC2626 !important;
        box-shadow: 0 6px 25px rgba(239,68,68,0.7) !important;
        transform: translateY(-2px) !important;
        border-color: #B91C1C !important;
    }
    
    .css-1d391kg {color: #EF4444 !important;}
    .stSidebar h2 {color: #EF4444 !important;}
    .stSuccess {background-color: rgba(239,68,68,0.2) !important; border-left: 4px solid #EF4444 !important;}
    .stInfo {background-color: rgba(239,68,68,0.1) !important; border-left: 4px solid #EF4444 !important;}
    .github-link {color: #F87171 !important; text-decoration: none !important; font-weight: bold;}
    .github-link:hover {color: #EF4444 !important; text-decoration: underline !important;}
    </style>
    """, unsafe_allow_html=True)

    # Session State
    for key in ['history', 'vector_store', 'current_q', 'current_a', 'show_workflow', 'ready_for_new']:
        if key not in st.session_state:
            st.session_state[key] = "" if key in ['current_q', 'current_a'] else [] if key == 'history' else False if key == 'show_workflow' else True

    
    st.markdown('<h1 class="header">PDF-BOT</h1>', unsafe_allow_html=True)
    
    # 🔥 Workflow toggle
    if st.button("🔬 How It Works", key="workflow_btn", help="View RAG pipeline"):
        st.session_state.show_workflow = not st.session_state.show_workflow

    if st.session_state.show_workflow:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.15); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #EF4444; margin: 1rem 0;'>
        <h3 style='color: #EF4444; margin-top: 0;'>✨ RAG Pipeline</h3>
        <ol style='font-size: 1.1rem; line-height: 1.6; color: #F3F4F6;'>
            <li><b>📄 PDF → Text:</b> Extract raw text from your PDF</li>
            <li><b>🔤 Chunking:</b> Split into 500-char pieces (50 overlap)</li>
            <li><b>🧠 Embeddings:</b> all-MiniLM-L6-v2 → 384D vectors</li>
            <li><b>🏪 FAISS:</b> Index vectors for lightning search</li>
            <li><b>❓ Query:</b> Your question → find Top-2 similar chunks</li>
            <li><b>⚡ Groq:</b> Llama 3.1 (8B) generates answer with context</li>
            <li><b>🎯 Result:</b> Precise answer in &lt;2 seconds!</b></li>
        </ol>
        <p style='font-style: italic; color: #FCA5A5; font-size: 0.95rem;'>
        Powered by <b>FAISS</b> + <b>Groq</b> + <b>HuggingFace Embeddings</b>
        </p>
        <p style='margin-top: 1rem;'>
        For more information : 
        <a href='https://github.com/Agnik101/RAG_APP' target='_blank' class='github-link'>
        click here!
        </a>
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### I'm Ready!")

    # Quick action buttons - Only work if ready for new input
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Key Points", key="keypoints") and st.session_state.ready_for_new:
            st.session_state.current_q = "List 5-8 KEY POINTS from this document as bullets:"
            st.session_state.current_a = ""
            st.session_state.ready_for_new = False
            st.rerun()
    
    with col2:
        if st.button("📄 Summary", key="summary") and st.session_state.ready_for_new:
            st.session_state.current_q = "Summarize the ENTIRE document in 200-300 words:"
            st.session_state.current_a = ""
            st.session_state.ready_for_new = False
            st.rerun()

    # 🔥 MAIN CHAT DISPLAY - ONLY CURRENT CHAT VISIBLE
    if st.session_state.current_q:
        st.markdown("### 💬 Current Chat")
        col_q, col_a = st.columns([1, 2])
        with col_q:
            st.info(f"**Q:** {st.session_state.current_q}")
        with col_a:
            if st.session_state.current_a:
                st.success(f"**A:** {st.session_state.current_a}")
            else:
                st.info("**A:** ⏳ Generating answer...")

    # 🔥 NEW QUESTION INPUT - Only shown when ready
    if st.session_state.ready_for_new:
        st.markdown("### 🔍 Ask your question:")
        q = st.text_input("", value="", key="question_input", 
                         placeholder="Type your question here and press Enter...")
        
        # Process only when Enter is pressed (q changes from empty to filled)
        if q and q.strip():
            st.session_state.current_q = q.strip()
            st.session_state.current_a = ""
            st.session_state.ready_for_new = False
            st.rerun()
    else:
        # Show "Ready for Next" button after chat completion
        if st.button("➡️ Ready for Next Question", key="next_question", help="Clear chat and ask new question"):
            st.session_state.current_q = ""
            st.session_state.current_a = ""
            st.session_state.ready_for_new = True
            st.rerun()

    # Sidebar
    api_key = None
    with st.sidebar:
        st.header("⚙️ Quick Setup")
        
        api_key = st.text_input("Groq Key", type="password", key="api_key")
        pdfs = st.file_uploader("PDFs", accept_multiple_files=True, type="pdf", key="pdfs")
        
        if st.button("🚀 Process", key="process"):
            if pdfs and api_key:
                with st.spinner("Processing please wait.."):
                    text = get_pdf_text(pdfs)
                    st.session_state.vector_store = get_vector_store(text)
                    st.success("✅ Ready!")
                    gc.collect()  
            else:
                st.error("❌ Please add PDFs and Groq API key first!")

        # 🔥 RECENT CHATS - Last 3 only
        st.markdown("### 📜 Recent Chats")
        if st.session_state.history:
            for h in st.session_state.history[-3:]:
                st.caption(f"• {h[:80]}...")
        else:
            st.caption("💭 No chats yet...")

    # 🔥 PROCESS QUESTION - Only when there's a question but no answer yet
    if st.session_state.current_q and not st.session_state.current_a:
        has_docs = 'vector_store' in st.session_state and st.session_state.vector_store
        has_key = bool(api_key and api_key.strip())
        
        if not has_docs or not has_key:
            st.warning("⚠️ Please add PDFs + Groq API key and click 'Process' first!")
        else:
            with st.spinner("🔮 Thinking..."):
                answer = get_groq_response(st.session_state.current_q, st.session_state.vector_store, api_key)
                
                # Update current chat
                st.session_state.current_a = answer
                
                # Add to history (keep only last 3)
                history_entry = f"Q: {st.session_state.current_q[:50]} | A: {answer[:50]}..."
                if history_entry not in st.session_state.history:
                    st.session_state.history.append(history_entry)
                    if len(st.session_state.history) > 3:
                        st.session_state.history = st.session_state.history[-3:]
                
                st.session_state.ready_for_new = False  # Stay in chat mode
            st.rerun()

if __name__ == "__main__":
    main()
