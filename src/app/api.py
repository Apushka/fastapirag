from contextlib import asynccontextmanager
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastapi import FastAPI, Depends, HTTPException
from src.settings import settings
from collections.abc import AsyncIterator
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
)
from pathlib import Path
import hashlib
from src.app.graph.builder import builder
from langgraph.checkpoint.memory import InMemorySaver
from src.app.clients import ollama, qdrant
from langgraph.graph.state import CompiledStateGraph
from src.app.graph.state import State
from src.app.schemas import AskRequest, AskResponse, HealthResponse
from langchain.messages import HumanMessage
import httpx
from ollama import ResponseError
import logging

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agent")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if await qdrant.collection_exists(collection_name=settings.collection_name):
        await qdrant.delete_collection(settings.collection_name)

    await qdrant.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_vector_size, distance=Distance.COSINE
        ),
    )

    docs_dir = Path(__file__).parent.parent / "docs"
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    all_points = []
    for md_file in docs_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        docs = splitter.split_text(content)

        for doc in docs:
            text = doc.page_content
            if not text.strip():
                continue

            response = await ollama.embed(model=settings.embedding_model, input=[text])

            all_points.append(
                PointStruct(
                    id=hashlib.md5(text.encode()).hexdigest(),
                    vector=response["embeddings"][0],
                    payload={
                        "text": text,
                        "source": md_file.name,
                        **doc.metadata,  # ← Header 1, Header 2, ...
                    },
                )
            )

    if all_points:
        await qdrant.upsert(collection_name=settings.collection_name, points=all_points)
        logger.info(f"Загружено {len(all_points)} чанков в бд")

    graph = builder.compile(checkpointer=InMemorySaver())
    app.state.graph = graph

    yield

    await qdrant.delete_collection(collection_name=settings.collection_name)
    await qdrant.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fast API RAG",
        description="Fast API RAG system",
        version="0.1.0",
        lifespan=lifespan,
    )

    def graph_dep() -> CompiledStateGraph[State]:
        compiled_graph = getattr(app.state, "graph", None)
        if compiled_graph is None:
            raise HTTPException(status_code=503, detail="Граф не готов")
        return compiled_graph

    @app.get("/health")
    async def check_health(graph: CompiledStateGraph[State] = Depends(graph_dep)):
        status = "OK"
        message = "Готов к работе"

        try:
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Верни строку 'ОК'. Ничего другого не возвращай"
                        )
                    ]
                },
                config={
                    "configurable": {"thread_id": "0"},
                },
            )
        except (ResponseError, httpx.ConnectError, Exception) as e:
            if isinstance(e, ResponseError):
                message = f"Ошибка LLM: {e}"
            if isinstance(e, httpx.ConnectError):
                message = "LLM недоступен"
            if isinstance(e, Exception):
                message = f"Ошибка: {e}"
            status = "FAIL"

        return HealthResponse(status=status, message=message)

    @app.post("/ask")
    async def ask(
        payload: AskRequest, graph: CompiledStateGraph[State] = Depends(graph_dep)
    ) -> AskResponse:
        prompt_hash = hashlib.sha256(payload.query.encode()).hexdigest()[:8]
        logger.info(
            f"Промпт пользователя {payload.user_id} (хэш: {prompt_hash}) отправлен в LLM"
        )

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=payload.query)]},
            config={
                "configurable": {"thread_id": str(payload.user_id)},
            },
        )
        return AskResponse(response=result["messages"][-1].content)

    return app


app = create_app()
