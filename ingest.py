import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ConfigError, StoreSettings, load_store_settings

logger = logging.getLogger("rag")

EXTENSIONS = {".txt", ".md"}


class IngestError(Exception):
    pass


class EmbeddingMismatchError(Exception):
    pass


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.is_dir():
        raise IngestError(f"No existe la carpeta de documentos: {data_dir}")
    archivos = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not archivos:
        raise IngestError(f"No hay archivos .txt ni .md en {data_dir}")
    documentos = []
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8").strip()
        if texto:
            documentos.append(Document(page_content=texto, metadata={"fuente": archivo.name}))
    if not documentos:
        raise IngestError(f"Todos los archivos de {data_dir} están vacíos")
    return documentos


def build_splitter(
    chunk_size: int, chunk_overlap: int, length_function: Callable[[str], int]
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function
    )


def split_documents(documentos: list[Document], splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    fragmentos = []
    for documento in documentos:
        partes = splitter.split_text(documento.page_content)
        for numero, parte in enumerate(partes):
            metadata = {"fuente": documento.metadata["fuente"], "fragmento": numero}
            fragmentos.append(Document(page_content=parte, metadata=metadata))
    return fragmentos


def open_collection(client: chromadb.ClientAPI, settings: StoreSettings, embeddings) -> chromadb.Collection:
    coleccion = client.get_or_create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": embeddings.model_name},
    )
    guardado = (coleccion.metadata or {}).get("embedding_model")
    if guardado != embeddings.model_name:
        raise EmbeddingMismatchError(
            f"La base se armó con el modelo de embeddings {guardado!r} y ahora se quiere usar {embeddings.model_name!r}. "
            "Con modelos distintos las distancias no tienen sentido: reindexá con --force o usá el mismo modelo."
        )
    return coleccion


def open_store(settings: StoreSettings, embeddings: Embeddings) -> Chroma:
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    open_collection(client, settings, embeddings)
    return Chroma(client=client, collection_name=settings.collection, embedding_function=embeddings)


def ingest(
    settings: StoreSettings,
    embeddings: Embeddings,
    force: bool = False,
    length_function: Callable[[str], int] | None = None,
) -> Chroma:
    length_function = length_function or getattr(embeddings, "count_tokens", None)
    if length_function is None:
        raise IngestError("Los embeddings no saben contar tokens: pasá length_function")
    limit = getattr(embeddings, "max_tokens", None)
    if limit and settings.chunk_size > limit - 2:
        raise IngestError(
            f"chunk_size {settings.chunk_size} supera el límite efectivo del modelo de embeddings "
            f"({limit} tokens, menos 2 especiales): usá {limit - 2} o menos"
        )
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.vectorstore_dir))
    exists = settings.collection in {getattr(c, "name", c) for c in client.list_collections()}
    if exists and not force:
        coleccion = open_collection(client, settings, embeddings)
        existentes = coleccion.count()
        if existentes > 0:
            logger.info("La base ya tiene %s fragmentos: no se vuelve a indexar (usar --force para reindexar)", existentes)
            return Chroma(client=client, collection_name=settings.collection, embedding_function=embeddings)
    inicio = time.perf_counter()
    documentos = load_documents(settings.data_dir)
    splitter = build_splitter(settings.chunk_size, settings.chunk_overlap, length_function)
    fragmentos = split_documents(documentos, splitter)
    ids = [f"{f.metadata['fuente']}:{f.metadata['fragmento']}" for f in fragmentos]
    textos = [f.page_content for f in fragmentos]
    vectores = embeddings.embed_documents(textos)
    if force and exists:
        client.delete_collection(settings.collection)
        logger.info("Se borró la colección %s para reindexar", settings.collection)
    coleccion = open_collection(client, settings, embeddings)
    coleccion.add(ids=ids, embeddings=vectores, documents=textos, metadatas=[f.metadata for f in fragmentos])
    logger.info(
        "Indexados %s fragmentos de %s archivos en %.1f s",
        len(fragmentos),
        len(documentos),
        time.perf_counter() - inicio,
    )
    return Chroma(client=client, collection_name=settings.collection, embedding_function=embeddings)


def main() -> None:
    from embeddings import build_embeddings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force = "--force" in sys.argv[1:]
    try:
        settings = load_store_settings()
        ingest(settings, build_embeddings(settings), force=force)
    except (ConfigError, IngestError, EmbeddingMismatchError) as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
