import streamlit as st
import os
from rag_core import RAGSystem

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="RAG Document Search", page_icon="🔍")

# --- HEADER ---
st.title("🔍 Document Search & Summarization (RAG)")
st.markdown("This tool searches a local knowledge base and answers questions using AI.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password", key="api_key_input")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter your Gemini API Key to enable summarization.")

    st.markdown("---")
    st.markdown("### Knowledge Base")
    st.text("Knowledge Base Files:")
    try:
        data_files = os.listdir(os.path.join(os.path.dirname(__file__), 'data'))
        for f in data_files:
            st.code(f)
    except:
        st.error("Data directory not found!")

# --- INITIALIZE RAG ---
# We use @st.cache_resource so we don't reload the model on every interaction
@st.cache_resource
def get_rag_system():
    rag = RAGSystem(use_llm=True)
    rag.load_documents()
    rag.build_index()
    return rag

try:
    with st.spinner("Loading specific AI models..."):
        rag = get_rag_system()
    st.success("System Ready!")
except Exception as e:
    st.error(f"Failed to initialize RAG: {e}")
    st.stop()

# --- MAIN INTERFACE ---
query = st.text_input("Ask a question about AI, Transformers, or RAG:")

if st.button("Search & Answer") or query:
    if not query:
        st.info("Please enter a question.")
    else:
        # 1. Retrieval
        with st.status("Searching knowledge base...", expanded=True) as status:
            st.write("Encoding query...")
            results = rag.search(query, top_k=3)
            st.write("Found relevant documents!")
            status.update(label="Retrieval Complete", state="complete", expanded=False)

        # Show Retrieved Context
        with st.expander("View Retrieved Context (Source Facts)"):
            for i, res in enumerate(results):
                st.info(f"**Chunk {i+1}**: {res}")

        # 2. Generation
        st.subheader("🤖 AI Answer")
        if not os.getenv("GOOGLE_API_KEY"):
             st.error("No Google API Key found. showing retrieval only.")
        else:
            with st.spinner("Generating summary..."):
                answer = rag.summarize(query, results)
                st.write(answer)

