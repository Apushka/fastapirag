from pydantic import BaseModel, Field
from typing import Literal


class AskRequest(BaseModel):
    user_id: int = Field(..., description="id пользователя")
    query: str = Field(..., description="Запрос пользователя")


class AskResponse(BaseModel):
    response: str = Field(..., description="Ответ")


class HealthResponse(BaseModel):
    status: Literal["OK", "FAIL"]
    message: str | None = Field(default=None, description="Описание статуса")
