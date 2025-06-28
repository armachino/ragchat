from core.agent import RagAgent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()


def run():
    """
    Main function to run the RagAgent.
    """
    # You can either download the HuggingFace model or use the already downloaded model from './models/hf_model'
    embedding_model = HuggingFaceEmbeddings(model_name="./models/hf_model")
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    vector_store = Chroma(
        collection_name="chatbot_collection",
        embedding_function=embedding_model,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    agent = RagAgent(
        llm_model=llm,
        vector_store=vector_store,
        docs_path="./docs",
    )
    pprint(agent.run("what is the  Hierarchy of grey wolf?"))  # Example question


if __name__ == "__main__":
    run()
