import asyncio
import sys
import time

import pytest
from fakes import FakeChat, no_lo_se, reply, words
from langchain_core.runnables import RunnableLambda

import ejemplo
from config import ConfigError
from ingest import ingest
from schemas import RespuestaRAG


@pytest.fixture
def preparar(monkeypatch, store_settings, embeddings, store):
    monkeypatch.setattr(ejemplo, "load_store_settings", lambda: store_settings)
    monkeypatch.setattr(ejemplo, "build_embeddings", lambda settings: embeddings)
    monkeypatch.setattr(ejemplo, "ingest", lambda settings, emb: ingest(settings, emb, length_function=words))

    def usar(chain, *nombres):
        monkeypatch.setattr(ejemplo, "get_default_chain", lambda: chain)
        monkeypatch.setattr(sys, "argv", ["ejemplo.py", *nombres])

    return usar


async def test_nombre_de_pregunta_desconocido(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ejemplo.py", "nada"])
    await ejemplo.main()
    salida = capsys.readouterr().out
    assert "Pregunta desconocida: nada" in salida
    assert "con-respuesta" in salida and "trampa-precio" in salida


async def test_pregunta_con_respuesta_imprime_el_json_con_referencias(preparar, capsys):
    from rag import build_chain

    preparar(build_chain(FakeChat(responses=[reply()]), 2), "con-respuesta")
    await ejemplo.main()
    salida = capsys.readouterr().out
    assert "=== con-respuesta ===" in salida
    assert "Pregunta: ¿Qué diferencia hay entre invoke, batch y stream?" in salida
    assert '"fuente": "chroma.md"' in salida


async def test_pregunta_trampa_imprime_no_lo_se_sin_referencias(preparar, capsys):
    from rag import build_chain

    preparar(build_chain(FakeChat(responses=[no_lo_se()]), 2), "trampa-precio")
    await ejemplo.main()
    salida = capsys.readouterr().out
    assert '"respuesta": "No lo sé"' in salida
    assert '"referencias": []' in salida


async def test_una_consulta_fallida_no_corta_las_siguientes(preparar, capsys):
    llamadas = []

    def inestable(entrada):
        llamadas.append(entrada)
        if len(llamadas) == 1:
            raise ConnectionError("sin red")
        return RespuestaRAG(respuesta="No lo sé")

    preparar(RunnableLambda(inestable), "con-respuesta", "trampa-precio")
    await ejemplo.main()
    salida = capsys.readouterr().out
    assert "Falló la consulta: ConnectionError: sin red" in salida
    assert '"respuesta": "No lo sé"' in salida
    assert len(llamadas) == 2


async def test_respuesta_invalida_se_informa_y_sigue(preparar, capsys):
    from rag import build_chain

    preparar(build_chain(FakeChat(responses=[reply(referencias=[])] * 3), 2), "con-respuesta")
    await ejemplo.main()
    assert "No se pudo validar la respuesta: la respuesta no cita ninguna fuente" in capsys.readouterr().out


def test_run_informa_un_error_de_configuracion_y_devuelve_1(monkeypatch, capsys):
    def falla():
        raise ConfigError("Falta ANTHROPIC_API_KEY en el .env")

    monkeypatch.setattr(ejemplo, "get_default_chain", falla)
    monkeypatch.setattr(sys, "argv", ["ejemplo.py", "con-respuesta"])
    assert ejemplo.run() == 1
    assert "Falta ANTHROPIC_API_KEY" in capsys.readouterr().out


async def test_arranque_no_bloquea_el_event_loop(preparar, monkeypatch, store):
    from rag import build_chain

    preparar(build_chain(FakeChat(responses=[reply()]), 2), "con-respuesta")
    ticks = []
    done = asyncio.Event()

    def slow_ingest(settings, embeddings):
        time.sleep(0.3)
        return store

    async def ticker():
        while not done.is_set():
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)
        ticks.append(time.perf_counter())

    monkeypatch.setattr(ejemplo, "ingest", slow_ingest)
    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        await ejemplo.main()
    finally:
        done.set()
        await task
    assert len(ticks) > 10
    assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.15
