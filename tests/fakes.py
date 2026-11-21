import hashlib
import json

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from schemas import normalizar


DOCUMENTOS = {
    "chroma.md": "ChromaDB guarda los vectores en una carpeta local persistente para no reindexar los documentos.",
    "embeddings.txt": "Un embedding es un vector numérico que representa el significado semántico de un texto.",
    "reintentos.md": "El reintento con espera exponencial duplica la espera después de cada intento fallido.",
    "plantas.txt": "La fotosíntesis convierte la luz solar en energía química dentro de las hojas.",
}


def words(texto: str) -> int:
    return len(texto.split())


class FakeEmbeddings(Embeddings):
    def __init__(self, model_name: str = "fake-embeddings", max_tokens: int | None = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.document_calls = 0
        self.query_calls = 0

    def count_tokens(self, text: str) -> int:
        return words(text)

    def vector(self, texto: str) -> list[float]:
        v = np.zeros(128)
        for palabra in normalizar(texto).split():
            v[int(hashlib.md5(palabra.encode()).hexdigest(), 16) % 128] += 1
        norma = np.linalg.norm(v)
        return (v / norma if norma else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self.vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self.vector(text)


class FakeChat(FakeMessagesListChatModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class FailingChat(FakeChat):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise ConnectionError("sin red")


class ContextChat(FakeMessagesListChatModel):
    calls: int = 0
    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        humano = messages[-1].content
        self.prompts.append(humano)
        contexto, pregunta = humano.split("\n\nPREGUNTA: ")
        palabras = [p for p in normalizar(pregunta).replace("?", "").split() if len(p) > 5]
        for bloque in contexto.removeprefix("CONTEXTO:\n").split("\n\n"):
            if any(p in normalizar(bloque) for p in palabras):
                fuente = bloque.split("Fuente: ")[1].split("\n")[0]
                cuerpo = bloque.split("\n", 1)[1]
                contenido = {
                    "respuesta": f"Según {fuente}: {cuerpo[:60]}",
                    "referencias": [{"fuente": fuente, "fragmento": cuerpo[:40]}],
                }
                break
        else:
            contenido = {"respuesta": "No lo sé", "referencias": []}
        mensaje = AIMessage(content=json.dumps(contenido, ensure_ascii=False))
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def reply(respuesta="La base se guarda en una carpeta local.", fuente="chroma.md", stop_reason="end_turn", **cambios):
    contenido = {
        "respuesta": respuesta,
        "referencias": [{"fuente": fuente, "fragmento": "una carpeta local persistente"}],
    }
    contenido.update(cambios)
    return AIMessage(content=json.dumps(contenido, ensure_ascii=False), response_metadata={"stop_reason": stop_reason})


def no_lo_se():
    return AIMessage(content=json.dumps({"respuesta": "No lo sé", "referencias": []}), response_metadata={"stop_reason": "end_turn"})


def text_reply(texto: str, stop_reason: str = "end_turn") -> AIMessage:
    return AIMessage(content=texto, response_metadata={"stop_reason": stop_reason})
