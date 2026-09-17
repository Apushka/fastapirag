from pydantic import BaseModel, Field
from typing import Annotated
from langchain_core.messages import BaseMessage
import operator


class State(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add]
    task_id: str = Field(default="")
