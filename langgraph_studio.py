from core.agent import RagAgent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma


from dotenv import load_dotenv

load_dotenv()

"""
Main function to run the RagAgent.
"""
embedding_model = HuggingFaceEmbeddings(model_name="./models/hf_model")
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

agent = RagAgent(
    llm_model=llm,
    vector_store=vector_store,
    docs_path="./docs",
)

# 👇 LangGraph Studio expects this variable
graph = agent.graph
