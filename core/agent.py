from core.utils.loader import get_loader_from_dir
from core.schema import RagState, InputState, OutputState

from langgraph.graph import StateGraph, START, END

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain.prompts import PromptTemplate


class RagAgent:
    """
    A state graph for a RAG agent that uses a chat prompt template and a regex parser.
    """

    def __init__(
        self,
        llm_model: BaseChatModel,
        vector_store: VectorStore,
        docs_path: str,
        prompt_template: PromptTemplate,
    ):
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.vector_store = vector_store

        # Load documents from a URL
        # docs = get_loader_from_url(
        #     "https://lilianweng.github.io/posts/2023-06-23-agent/"
        # )
        docs = get_loader_from_dir(docs_path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        # Use the vector_store initialized with HuggingFaceEmbeddings
        _ = self.vector_store.add_documents(documents=all_splits)

        # Define prompt for question-answering
        builder = StateGraph(
            RagState, input_schema=InputState, output_schema=OutputState
        )
        # Define the nodes
        builder.add_node("retrieve", self.retrieve)  # type: ignore
        builder.add_node("generate", self.generate)
        # Define the edges
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        self.graph = builder.compile()

    # Define application steps
    def retrieve(self, state: InputState) -> RagState:
        # Use the vector_store initialized with HuggingFaceEmbeddings
        retrieved_docs = self.vector_store.similarity_search(state.question)
        return RagState(
            question=state.question,
            context=retrieved_docs,
            answer="",  # Initially empty, will be filled in later
        )

    def generate(self, state: RagState) -> OutputState:
        docs_content = "\n\n".join(doc.page_content for doc in state.context)
        messages = self.prompt_template.invoke(
            {"question": state.question, "context": docs_content}
        )
        response = self.llm_model.invoke(messages)
        return OutputState(answer=response.content.__str__())
        # return {"answer": response.content}

    def run(self, question: str):
        input_state_dict = {
            "question": question,
        }

        input_state = InputState.model_validate(input_state_dict)
        response = self.graph.invoke(
            input_state.model_dump()  # Convert InputState to dict for invocation # type: ignore
        )
        print(response["answer"])


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
        prompt_template=prompt,
    )
    agent.run("What is his internet status?")  # Example question
