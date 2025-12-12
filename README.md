# Ever Quint Interview Assignments

This repository contains the completed solutions for the Ever Quint interview assignments (Gen AI Engineer). 

## 📂 Project Structure

This project is organized into four main directories, corresponding to the assigned tasks:

1.  **`AI_Engineer_CIFAR10/`** (Mandatory) - Image Classification using PyTorch.
2.  **`Algorithm_Max_Profit/`** (Mandatory) - Dynamic Programming solution for maximizing earnings.
3.  **`Algorithm_Water_Tank/`** (Mandatory) - Web-based visualization of the Water Tank problem.
4.  **`Gen_AI_Engineer_RAG/`** (Bonus) - RAG System with Gemini 2.5 Flash and Streamlit UI.
5.  **`Gen_AI_Engineer_Reasoning/`** (Bonus) - Multistep Reasoning Agent (Planner-Executor-Verifier).

---

## 🛠️ Prerequisites

-   Python 3.8+
-   Pip (Python Package Manager)
-   A Web Browser

---

## 1️⃣ Gen AI Engineer - RAG (Bonus)

A Retrieval-Augmented Generation (RAG) system that answers questions based on a local knowledge base.

**Key Features:**
-   **Model:** Google Gemini 2.5 Flash (via `google-generativeai`).
-   **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformer).
-   **Vector DB:** FAISS (Facebook AI Similarity Search).
-   **Interface:** Streamlit Web App (Bonus Requirement).

### Setup & Run
1.  Navigate to the directory:
    ```bash
    cd Gen_AI_Engineer_RAG
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Web Interface:**
    ```bash
    streamlit run app.py
    ```
    *(Or run `Run_RAG_App.bat` from the root directory)*
4.  **Run in Terminal (CLI Mode):**
    ```bash
    python rag_core.py
    ```

> **Note:** You will need a **Google Gemini API Key** to enable the summarization feature (Input it in the UI or CLI prompt).

---

## 2️⃣ AI Engineer - CIFAR-10 (Mandatory)

A Convolutional Neural Network (CNN) built from scratch to classify CIFAR-10 images. Includes a **Residual Block** improvement (Option A).

### Setup & Run
1.  Navigate to the directory:
    ```bash
    cd AI_Engineer_CIFAR10
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the Model:**
    ```bash
    python train.py
    ```
    *   Downloads dataset automatically.
    *   Trains for 1 Epoch (for demo purposes) on CPU/GPU.
    *   Saves model to `cifar_net.pth`.
    *   Generates `training_loss.png` and `misclassified.png`.

---

## 3️⃣ Algorithm - Max Profit (Mandatory)

A solution to maximize property earnings based on build time and operational duration.

### Run
1.  Run the Python script:
    ```bash
    python Algorithm_Max_Profit/solution.py
    ```
2.  **Output:** Prints the optimal building mix and total earnings for the test cases (Time 7, 8, 13).

---

## 4️⃣ Algorithm - Water Tank (Mandatory)

A visualization of the "Trapping Rain Water" problem using HTML, CSS, and Vanilla JavaScript.

### Run
1.  Navigate to `Algorithm_Water_Tank/`.
2.  Open **`index.html`** in any web browser.

---

## 5️⃣ Gen AI Engineer - Multistep Reasoning (Bonus)

A Reasoning Agent that solves structured problems using a **Planner -> Executor -> Verifier** loop.

**Key Features:**
-   **Architecture:** Separates planning, math execution, and self-verification.
-   **Self-Correction:** If the Verifier spots a mistake, the agent retries (up to 3 times) with feedback.
-   **Mock Mode:** If `GOOGLE_API_KEY` is missing, it falls back to a structural mock (verifying the loop logic).

**Setup:**
```bash
# Set your Google API Key (Required for live Gemini usage)
set GOOGLE_API_KEY=your_key_here
```

**Run Evaluation:**
```bash
python Gen_AI_Engineer_Reasoning/evaluate.py
```

**Run Interactive Agent:**
```bash
python Gen_AI_Engineer_Reasoning/agent.py


# Use $env:GOOGLE_API_KEY="***********"; python Gen_AI_Engineer_Reasoning/evaluate.py (in the terminal)
```

---

## 📈 Project Report & Design Decisions

### 1. Gen AI - RAG System
-   **Methodology:**
    -   **Ingestion:** Text files are loaded and split into chunks (lines) to maintain granularity.
    -   **Embeddings:** Used `all-MiniLM-L6-v2` (384d) for its speed and adequate performance for semantic search.
    -   **Vector DB:** `FAISS` was chosen for efficient, local, in-memory similarity search (L2 distance).
    -   **Summarization:** Switched to **Google Gemini 2.5 Flash** for highly efficient, low-latency summarization of retrieved contexts.
-   **Challenges:**
    -   *Challenge:* Handling API keys securely in a public repo. *Solution:* Used environment variables and `st.text_input` (password type) in the UI.
    -   *Challenge:* Latency. *Solution:* Used "Flash" model variant and cached the `RAGSystem` resource in Streamlit (`@st.cache_resource`).

### 2. AI Engineer - CIFAR-10
-   **Preprocessing:** Standard normalization `(0.5, 0.5, 0.5)` helps the CNN converge faster. Random Horizontal Flip and Crop added for robustness (Data Augmentation).
-   **Architecture:** Built a custom CNN with **Residual Blocks** (Option A). The skip connections in ResNet help prevent vanishing gradients, allowing deeper networks to train effectively.
-   **Evaluation:** The model is evaluated on the 10,000-image test set. A confusion matrix approach (visualizing misclassified images) helps identify where the model struggles (e.g., Cat vs Dog).

### 3. Gen AI - Multistep Reasoning
-   **Architecture:** Implemented the "Reasoning Agent" pattern (Planner/Executor/Verifier) to decouple logic steps.
-   **Verification:** The key innovation is the retry loop—Gemini is asked to "verify" its own output. If verification fails (JSON status), it feeds the feedback back into the prompt context for the next attempt.
-   **Robustness:** Added a **Mock Mode** to ensure the agent's structural logic (state transitions, JSON parsing) can be verified even without an active API key in the generic CI/CD environment.

---
**Repository Link:** [https://github.com/AswathVishahan/assignment-everquint](https://github.com/AswathVishahan/assignment-everquint)

