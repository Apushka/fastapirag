from langchain.tools import tool, ToolRuntime
import uuid
from langgraph.types import Command
from langchain.messages import ToolMessage
from src.app.clients import qdrant, ollama
from src.settings import settings
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger("agent")


class SearchDocumentationInputSchema(BaseModel):
    query: str = Field(..., description="Запрос пользователя")

    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool(args_schema=SearchDocumentationInputSchema)
async def search_documentation(query: str, runtime: ToolRuntime) -> str:
    """Вызови, если нужна информация из векторного хранилища"""
    logger.info("Вызван инструмент поиска по докуменации")
    try:
        response = await ollama.embed(
            model=settings.embedding_model,
            input=[query],
        )

        vector = response["embeddings"][0]
        search_results = await qdrant.query_points(
            collection_name=settings.collection_name, query=vector, limit=3
        )
        context = "\n\n".join(s.payload["text"] for s in search_results.points)

        logger.info(f"Найдено {len(search_results.points)} релевантных чанков")
    except Exception as e:
        logger.error("Ошибка в search_documentation", extra={"error": str(e)})
        return "Не удалось выполнить поиск в документации."

    return ToolMessage(content=context, tool_call_id=runtime.tool_call_id)


class CreateTaskInput(BaseModel):
    description: str = Field(..., description="Описание для создания задачи.")
    runtime: ToolRuntime

    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool(args_schema=CreateTaskInput)
def create_task(description: str, runtime: ToolRuntime) -> Command[uuid.UUID]:
    """Вызови, если нужно создать задачу"""
    logger.info("Вызван инструмент создания новой задачи")
    try:
        task_id = uuid.uuid4()
        logger.info(f"Создана задача с task_id: {task_id}")
    except Exception as e:
        logger.error("Ошибка в create_task", extra={"error": str(e)})
        return "Произошла ошибка при создании новой задачи"
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=description,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "task_id": str(task_id),
        }
    )


class AddCommentInputSchema(BaseModel):
    task_id: uuid.UUID = Field(
        ..., description="task_id задачи, к которой нужно добавить комментарий"
    )
    comment: str = Field(..., description="Комментарий к задаче")
    runtime: ToolRuntime

    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool(args_schema=AddCommentInputSchema)
def add_comment(task_id: uuid.UUID, comment: str, runtime: ToolRuntime) -> str:
    """Вызови, если нужно добавить комментарий к задаче"""
    logger.info(f"Вызван инструмент добавления комментария '{comment}' к задаче {task_id}")
    try:
        logger.info(f"Создан комментарий '{comment}' для задачи {task_id}")
    except Exception as e:
        logger.error("Ошибка в add_comment", extra={"error": str(e)})
        return "Произшола ошибка при добавлении комментария"
    return ToolMessage(
        content=f"Addded the following comments to #{str(uuid)}:\n{comment}",
        tool_call_id=runtime.tool_call_id,
    )


tools = [search_documentation, create_task, add_comment]
tool_node = ToolNode(tools)
