import asyncio
import logging
import sys

from config import ConfigError, load_store_settings
from embeddings import build_embeddings
from ingest import EmbeddingMismatchError, IngestError, ingest
from rag import IncompleteOutputError, InvalidOutputError, get_default_chain, get_rag_response
from retriever import EmptyStoreError

PREGUNTAS = {
    "con-respuesta": "¿Qué diferencia hay entre invoke, batch y stream?",
    "reintentos": "¿Por qué el reintento envuelve la cadena entera y no solo el modelo?",
    "trampa-precio": "¿Cuánto cuesta el millón de tokens de Claude Haiku?",
    "trampa-despliegue": "¿Cómo se despliega un pipeline en Kubernetes?",
}


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    elegidas = sys.argv[1:] or list(PREGUNTAS)
    desconocidas = [n for n in elegidas if n not in PREGUNTAS]
    if desconocidas:
        print(f"Pregunta desconocida: {', '.join(desconocidas)}. Opciones: {', '.join(PREGUNTAS)}")
        return
    chain = await asyncio.to_thread(get_default_chain)
    settings = load_store_settings()
    embeddings = await asyncio.to_thread(build_embeddings, settings)
    store = await asyncio.to_thread(ingest, settings, embeddings)
    for nombre in elegidas:
        pregunta = PREGUNTAS[nombre]
        print(f"\n=== {nombre} ===\nPregunta: {pregunta}")
        try:
            respuesta = await get_rag_response(pregunta, store=store, chain=chain, top_k=settings.top_k)
        except (InvalidOutputError, IncompleteOutputError) as error:
            print(f"No se pudo validar la respuesta: {error}")
        except Exception as error:
            print(f"Falló la consulta: {type(error).__name__}: {error}")
        else:
            print(respuesta.model_dump_json(indent=2))


def run() -> int:
    try:
        asyncio.run(main())
    except (ConfigError, IngestError, EmbeddingMismatchError, EmptyStoreError) as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
