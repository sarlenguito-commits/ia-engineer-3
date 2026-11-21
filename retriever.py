import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger("rag")


class EmptyStoreError(Exception):
    pass


async def retrieve(store: Chroma, query: str, top_k: int) -> list[tuple[Document, float]]:
    resultados = await store.asimilarity_search_with_score(query, k=top_k)
    if not resultados:
        raise EmptyStoreError("La base vectorial no tiene fragmentos: corré ingest.py primero")
    similares = [(documento, 1.0 - distancia) for documento, distancia in resultados]
    logger.info(
        "Fragmentos recuperados: %s",
        ", ".join(f"{d.metadata['fuente']}#{d.metadata['fragmento']} ({s:.2f})" for d, s in similares),
    )
    return similares
