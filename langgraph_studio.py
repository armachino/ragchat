from core.agent import RagAgent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain import hub


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
prompt = hub.pull("rlm/rag-prompt")

agent = RagAgent(
    llm_model=llm,
    vector_store=vector_store,
    docs_path="./docs",
    prompt_template=prompt,
)

# 👇 LangGraph Studio expects this variable
graph=agent.graph
agent.run("Tell me about his Wi-fi Status in the village")  # Example question
