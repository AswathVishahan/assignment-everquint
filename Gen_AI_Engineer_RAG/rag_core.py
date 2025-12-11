import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- CONFIGURATION ---
# We use a small, fast model for embeddings.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class RAGSystem:
    def __init__(self, use_llm=True):
        """
        Initialize the RAG system.
        Loads the embedding model and prepares the index.
        """
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.documents = [] # Stores key-value pairs of (id, text_chunk)
        self.use_llm = use_llm
        
        # Check for Google API Key if needed
        if self.use_llm: # Keeping variable name generic or change to use_llm
            if not os.getenv("GOOGLE_API_KEY"):
                print("WARNING: GOOGLE_API_KEY not found in environment variables.")
                print("Summarization might fail unless you provide a key.")

    def load_documents(self):
        """
        Reads all .txt files from the data directory.
        Chunks them into sentences or paragraphs.
        """
        file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
        print(f"Found {len(file_paths)} documents in {DATA_DIR}")
        
        chunks = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple chunking by newline for this assignment. 
                # In production, use a recursive character splitter (e.g., LangChain).
                # We filter out empty lines.
                file_chunks = [line.strip() for line in content.split('\n') if line.strip()]
                chunks.extend(file_chunks)
        
        self.documents = chunks
        print(f"Total chunks created: {len(self.documents)}")
        return self.documents

    def build_index(self):
        """
        Encodes all document chunks and builds a FAISS index.
        """
        if not self.documents:
            print("No documents to index. Call load_documents() first.")
            return

        print("Encoding documents... (This may take a moment)")
        embeddings = self.encoder.encode(self.documents)
        
        # FAISS expects float32
        embeddings = np.array(embeddings).astype('float32')
        
        # Dimension of the embeddings (384 for MiniLM)
        dimension = embeddings.shape[1]
        
        # Create a simple Flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def search(self, query, top_k=3):
        """
        Searches the index for the most similar chunks to the query.
        """
        if not self.index:
            print("Index not built.")
            return []

        # Encode the query
        query_vector = self.encoder.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # -1 means not found (shouldn't happen in exact search)
                results.append(self.documents[idx])
        
        return results

    def summarize(self, query, retrieved_chunks):
        """
        Uses an LLM (Gemini) to summarize the answer based on retrieved chunks.
        """
        context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])
        
        prompt = f"""
You are a helpful AI assistant. Use the following context to answer the user's question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:
        """
        
        if self.use_llm:
            try:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error using Gemini: {e}. \n(Ensure you have set GOOGLE_API_KEY)"
        else:
            return "Local summarization is not fully implemented in this simple example.\nHowever, here are the retrieved facts:\n" + context

if __name__ == "__main__":
    # Simple CLI Test for RAG
    print("--- RAG System CLI Test ---")
    
    # Prompt for key if not present (Fixes user error)
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n[INFO] GOOGLE_API_KEY not found in environment.")
        key_input = input("Please paste your Google Gemini API Key: ").strip()
        if key_input:
            os.environ["GOOGLE_API_KEY"] = key_input
        else:
            print("[WARNING] No key provided. Summarization will fail.")

    rag = RAGSystem(use_llm=True)
    rag.load_documents()
    rag.build_index()
    
    print("\n--- RAG System Ready ---")
    while True:
        print("-" * 30)
        query = input("Ask a question (or type 'exit' to quit): ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        # Search
        results = rag.search(query)
        print("\nRetrieved Facts:")
        for i, r in enumerate(results):
            print(f"{i+1}. {r}")
            
        # Summarize
        print("\nAI Answer:")
        answer = rag.summarize(query, results)
        print(answer)
        print("-" * 30)
