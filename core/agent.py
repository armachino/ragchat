from core.utils.loader import get_loader_from_dir
from core.utils.get_entry_script import get_entry_script_name

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
# from pprint import pprint

# from langgraph.checkpoint.memory import MemorySaver


class RagAgent:
    def __init__(
        self,
        llm_model: BaseChatModel,
        vector_store: VectorStore,
        docs_path: str,
    ):
        self._vector_store = vector_store
        docs = get_loader_from_dir(docs_path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)

        # print("all_splits___",all_splits)
        # Index chunks
        # Use the vector_store initialized with HuggingFaceEmbeddings
        _ = self._vector_store.add_documents(documents=all_splits)

        # toolset = AgentToolset(vector_store=vector_store)
        tools = [self._get_retrieve_tool(self._vector_store)]

        # Check if running inside LangGraph Studio (entry ends with "main.py").
        # If so, use MemorySaver as checkpointer; otherwise, omit it.
        entry = get_entry_script_name()
        if entry and entry.endswith("main.py"):
            from langgraph.checkpoint.memory import MemorySaver
            # print("***main.py****")
            self._agent_executor = create_react_agent(
                llm_model, tools, checkpointer=MemorySaver()
            )
        else:
            # print("***langgraph studio****")
            self._agent_executor = create_react_agent(llm_model, tools)

    def _get_retrieve_tool(self, vector_store: VectorStore):
        @tool
        def retrieve(query: str) -> tuple[str, list[Document]]:
            """Retrieve documents from vector store based on a query."""
            if vector_store is None:
                raise ValueError("Vector store is not available.")

            # Retrieve documents using the passed vector_store
            retrieved_docs = vector_store.similarity_search(query,k=3)
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            # print(f"Retrieved documents: {serialized}")
            return serialized, retrieved_docs

        return retrieve

    @property
    def graph(self):
        # The getter method that returns the workflow
        return self._agent_executor

    def run(self, question: str, thread_id: str = "abc123") -> str:
        config = RunnableConfig(configurable={"thread_id": thread_id})
        # for event in agent_executor.stream(
        #     {"messages": [HumanMessage(content="Who is Armapopoli?")]},
        #     stream_mode="values",
        #     config=config,
        # ):
        #     event["messages"][-1].pretty_print()
        respone = self._agent_executor.invoke(
            {"messages": [HumanMessage(content=question)]}, config=config
        )
        # pprint(respone["messages"][-1].content)
        return respone["messages"][-1].content


if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chat_models import init_chat_model
    from langchain_chroma import Chroma
    from langchain import hub

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
        # prompt_template=prompt,
    )
    # agent.run("Tell me about the GWO")
