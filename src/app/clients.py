from qdrant_client import AsyncQdrantClient
from ollama import AsyncClient
from src.settings import settings

qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
ollama = AsyncClient(host=settings.ollama_host)
