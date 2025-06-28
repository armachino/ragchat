# RagChat 🧠💬

**RagChat** is a simple yet powerful PDF-based chatbot powered by Retrieval-Augmented Generation (RAG). It indexes documents from the `docs/` folder and uses LangGraph + LangChain to answer user queries based on that content.



## 📦 Features

- 🔍 PDF loading & chunking  
- 📚 Document indexing using ChromaDB  
- 💬 Context-aware Q&A via LangGraph agent  
- 📁 Modular code structure for maintainability  
- ⚡ Local Hugging Face embedding model (custom path under `models/`)



## 🛠 Tech Stack

- Python  
- LangGraph  
- LangChain  
- ChromaDB  
- Sentence Transformers (HuggingFace)



## 📂 Folder Structure

```

RagChat/
├── chroma\_langchain\_db/         # Chroma vector store
├── core/                        # Agent logic and utilities
│   ├── agent.py
│   ├── utils/
│   │   ├── loader.py
│   │   └── env.py
├── docs/                        # Source PDFs
│   └── GWO-output.pdf           # Sample document (article on GWO algorithms)             
├── environment.yml              # <-- Here is your Conda environment file
├── main.py                      # Main entry point
├── models/                      # local models
│   └── hf\_model/
├── notebooks/                   # Dev notebooks
├── langgraph\_studio.py         # Studio-compatible script
```
The docs/ folder includes a test document: GWO-output.pdf, which is an article about the Grey Wolf Optimizer (GWO) algorithm. You can replace or add more PDFs for custom use cases.


## ⚙️ Installation (with Conda)

```bash
# Step 1: Clone the repo
git clone https://github.com/yourname/RagChat.git
cd RagChat

# Step 2: Create and activate environment
conda env create -f environment.yml
conda activate ragchat-env
```

> You can either pass the path to a local Sentence Transformer model (e.g., under models/hf_model/) or provide an API URL to use a remote embedding service.


## 📄 .env Configuration

Create a `.env` file in the root directory to store environment variables securely. Below is a sample structure:

```env
# LangSmith for LangGraph / LangChain logging & traces
LANGSMITH_API_KEY=your_langsmith_api_key

# Background job settings (used for loop isolation or debugging)
BG_JOB_ISOLATED_LOOPS=true

# Google API keys (for tools like Google Search)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CLOUD_API_KEY=your_google_cloud_api_key

# Hugging Face model access (if using API)
HUGGING_FACE_API_KEY=your_huggingface_api_key

# Optional: Custom user agent for web requests
USER_AGENT=ragchat-agent

```


## 🚀 Running the Project

### Option 1: Standard Python

To run the chatbot normally:

```bash
python main.py
```

This will load the documents in `docs/`, index them into ChromaDB, and launch the RAG-based agent.



### Option 2: Using LangGraph Studio

LangGraph Studio allows you to visually explore and debug your RAG pipeline.

```bash
langgraph dev --allow-blocking
```

> Make sure you're in the root directory where `langgraph_studio.py` is located.

This auto-runs the graph defined in `langgraph_studio.py`, skipping any manual checkpointer.


## 🧠 Memory & Persistence

* When using `main.py`, memory is handled with `InMemorySaver()`.
* When using LangGraph Studio, built-in persistence is used automatically.
* This detection is automatic via the script name.



## 📝 License

MIT License



<!-- ## 👤 Author

**Arman Ranjbar**

For questions, contributions, or feedback — feel free to reach out or submit a PR.

```



Let me know if you'd like help generating a `requirements.txt` file, or adding deployment instructions (like with FastAPI or Streamlit).
``` -->
