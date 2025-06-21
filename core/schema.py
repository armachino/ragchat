from typing_extensions import List#, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel
# from typing_extensions import Literal


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    answer: str

class RagState(BaseModel):
    question: str
    context: List[Document]
    answer: str
