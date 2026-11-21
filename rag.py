import asyncio
import logging
import threading
import time
from functools import cache
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from pydantic import ValidationError

from config import StoreSettings, load_llm_settings, load_store_settings
from models import create_model
from retriever import retrieve
from schemas import NO_LO_SE, RespuestaRAG, explicar_errores, normalizar_cita

logger = logging.getLogger("rag")

STOP_REASONS_BY_LENGTH = {"max_tokens", "length"}
MAX_QUERY_LENGTH = 1000
DEFAULT_LOCK = threading.Lock()

SYSTEM_PROMPT = (
    "Sos un asistente técnico. Respondé solo basándote en el CONTEXTO proporcionado, sin usar conocimiento externo. "
    f"Si la respuesta no está en el contexto, respondé exactamente '{NO_LO_SE}' y no cites ninguna fuente. "
    "En 'referencias' poné, por cada dato que uses, el nombre del archivo (Fuente) y una frase corta copiada del contexto. "
    "{instrucciones_formato}"
)

HUMAN_PROMPT = "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"

PARSER = PydanticOutputParser(pydantic_object=RespuestaRAG)


class InvalidOutputError(Exception):
    pass


class IncompleteOutputError(Exception):
    pass


def build_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt.partial(instrucciones_formato=PARSER.get_format_instructions())


def format_context(resultados: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[{numero}] Fuente: {documento.metadata['fuente']}\n{documento.page_content}"
        for numero, (documento, _) in enumerate(resultados, start=1)
    )


def message_text(message: BaseMessage) -> str:
    contenido = message.content
    if isinstance(contenido, str):
        return contenido
    return "".join(b.get("text", "") for b in contenido if isinstance(b, dict) and b.get("type") == "text")


def log_call(inputs: dict) -> dict:
    logger.info("Llamando al modelo")
    return inputs


def check_response(inputs: dict) -> RespuestaRAG:
    mensaje = inputs["mensaje"]
    fuentes = inputs["fuentes"]
    metadata = getattr(mensaje, "response_metadata", None) or {}
    reason = metadata.get("stop_reason") or metadata.get("finish_reason")
    if reason in STOP_REASONS_BY_LENGTH:
        logger.warning("La respuesta se cortó por falta de tokens (%s); se reintenta si quedan intentos", reason)
        raise IncompleteOutputError(f"La respuesta se cortó por falta de tokens ({reason}); subí LLM_MAX_TOKENS")
    try:
        respuesta = PARSER.parse(message_text(mensaje))
    except OutputParserException as error:
        causa = error.__cause__ or error.__context__
        if isinstance(causa, ValidationError):
            detalle = explicar_errores(causa)
        else:
            detalle = "la respuesta del modelo no es un JSON válido con el formato pedido"
        logger.debug("Detalle técnico del error: %s", error)
        logger.warning("La respuesta no validó: %s; se reintenta si quedan intentos", detalle)
        raise InvalidOutputError(detalle) from error
    problema = None
    if respuesta.sin_respuesta:
        if respuesta.referencias:
            problema = f"una respuesta '{NO_LO_SE}' no puede traer referencias"
    elif not respuesta.referencias:
        problema = "la respuesta no cita ninguna fuente"
    else:
        inventadas = sorted({r.fuente for r in respuesta.referencias} - set(fuentes))
        if inventadas:
            problema = f"las referencias {', '.join(inventadas)} no están entre las fuentes recuperadas ({', '.join(fuentes)})"
        else:
            for referencia in respuesta.referencias:
                if normalizar_cita(referencia.fragmento) not in inputs["textos"][referencia.fuente]:
                    problema = (
                        f"la referencia a {referencia.fuente} cita un fragmento que no aparece en lo recuperado: "
                        f"{referencia.fragmento[:60]!r}"
                    )
                    break
    if problema:
        logger.warning("La respuesta no validó: %s; se reintenta si quedan intentos", problema)
        raise InvalidOutputError(problema)
    logger.info("Respuesta validada")
    return respuesta


def build_chain(llm: Runnable, retries: int = 2) -> Runnable:
    async def timed_model(prompt, config):
        inicio = time.perf_counter()
        try:
            return await llm.ainvoke(prompt, config=config)
        finally:
            logger.info("Llamada al modelo: %.0f ms", (time.perf_counter() - inicio) * 1000)

    chain = (
        RunnableLambda(log_call)
        | RunnableParallel(
            mensaje=build_prompt() | RunnableLambda(timed_model),
            fuentes=itemgetter("fuentes"),
            textos=itemgetter("textos"),
        )
        | RunnableLambda(check_response)
    )
    return chain.with_retry(
        retry_if_exception_type=(InvalidOutputError, IncompleteOutputError),
        wait_exponential_jitter=False,
        stop_after_attempt=retries + 1,
    )


def build_default_chain() -> Runnable:
    settings = load_llm_settings()
    model = create_model(
        settings.provider,
        settings.model,
        settings.api_key.get_secret_value(),
        settings.max_tokens,
        settings.timeout,
    )
    return build_chain(model, settings.max_retries)


@cache
def _get_default_chain() -> Runnable:
    return build_default_chain()


@cache
def _get_default_settings() -> StoreSettings:
    return load_store_settings()


@cache
def _get_default_store():
    from embeddings import build_embeddings
    from ingest import open_store

    settings = _get_default_settings()
    return open_store(settings, build_embeddings(settings))


def get_default_chain() -> Runnable:
    with DEFAULT_LOCK:
        return _get_default_chain()


def get_default_settings() -> StoreSettings:
    with DEFAULT_LOCK:
        return _get_default_settings()


def get_default_store():
    with DEFAULT_LOCK:
        return _get_default_store()


async def get_rag_response(query: str, store=None, chain: Runnable | None = None, top_k: int | None = None) -> RespuestaRAG:
    if not query or not query.strip():
        raise ValueError("La pregunta no puede estar vacía")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"La pregunta es demasiado larga: tiene {len(query)} caracteres y el máximo es {MAX_QUERY_LENGTH}")
    inicio = time.perf_counter()
    store = store or await asyncio.to_thread(get_default_store)
    chain = chain or await asyncio.to_thread(get_default_chain)
    if top_k is None:
        top_k = (await asyncio.to_thread(get_default_settings)).top_k
    inicio_recuperacion = time.perf_counter()
    resultados = await retrieve(store, query, top_k)
    tras_recuperar = time.perf_counter()
    fuentes = sorted({documento.metadata["fuente"] for documento, _ in resultados})
    textos = {
        fuente: normalizar_cita(" ".join(d.page_content for d, _ in resultados if d.metadata["fuente"] == fuente))
        for fuente in fuentes
    }
    respuesta = await chain.ainvoke(
        {"pregunta": query.strip(), "contexto": format_context(resultados), "fuentes": fuentes, "textos": textos}
    )
    fin = time.perf_counter()
    logger.info(
        "Tiempos: recuperación %.0f ms, modelo %.0f ms (con reintentos), total %.0f ms",
        (tras_recuperar - inicio_recuperacion) * 1000,
        (fin - tras_recuperar) * 1000,
        (fin - inicio) * 1000,
    )
    return respuesta
