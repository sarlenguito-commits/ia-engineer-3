import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fakes import ContextChat, FailingChat, FakeChat, no_lo_se, reply, text_reply

from rag import (
    MAX_QUERY_LENGTH,
    IncompleteOutputError,
    InvalidOutputError,
    build_chain,
    build_prompt,
    format_context,
    get_rag_response,
)
from retriever import retrieve
from schemas import RespuestaRAG


def chain_with(respuestas, retries=2, cls=FakeChat):
    modelo = cls(responses=respuestas)
    return build_chain(modelo, retries), modelo


async def test_pregunta_con_respuesta_en_los_documentos(store):
    chain, modelo = chain_with([reply()])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert isinstance(respuesta, RespuestaRAG)
    assert not respuesta.sin_respuesta
    assert respuesta.referencias[0].fuente == "chroma.md"
    assert modelo.calls == 1


async def test_pregunta_trampa_responde_no_lo_se_sin_referencias(store):
    modelo = ContextChat(responses=[])
    chain = build_chain(modelo, 2)
    respuesta = await get_rag_response("¿Cuál es la capital de Francia?", store, chain, top_k=3)
    assert respuesta.sin_respuesta
    assert respuesta.referencias == []


async def test_el_flujo_completo_con_contexto_real_cita_la_fuente_correcta(store):
    modelo = ContextChat(responses=[])
    chain = build_chain(modelo, 2)
    respuesta = await get_rag_response("¿Qué hace la fotosíntesis con la luz solar?", store, chain, top_k=3)
    assert not respuesta.sin_respuesta
    assert [r.fuente for r in respuesta.referencias] == ["plantas.txt"]


async def test_el_prompt_lleva_el_contexto_y_la_pregunta(store):
    modelo = ContextChat(responses=[])
    chain = build_chain(modelo, 2)
    await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    humano = modelo.prompts[-1]
    assert humano.startswith("CONTEXTO:\n[1] Fuente: ")
    assert "PREGUNTA: ¿Dónde guarda ChromaDB los vectores?" in humano
    assert humano.count("Fuente: ") == 3


async def test_el_prompt_de_sistema_pide_no_lo_se_y_trae_el_formato():
    mensajes = build_prompt().invoke({"contexto": "ctx", "pregunta": "p"}).to_messages()
    sistema = mensajes[0].content
    assert "No lo sé" in sistema
    assert "solo basándote en el CONTEXTO" in sistema
    assert '"respuesta"' in sistema and '"referencias"' in sistema


def test_format_context_numera_y_nombra_las_fuentes():
    from langchain_core.documents import Document

    resultados = [
        (Document(page_content="uno", metadata={"fuente": "a.md", "fragmento": 0}), 0.9),
        (Document(page_content="dos", metadata={"fuente": "b.txt", "fragmento": 0}), 0.8),
    ]
    assert format_context(resultados) == "[1] Fuente: a.md\nuno\n\n[2] Fuente: b.txt\ndos"


async def test_json_incompleto_sin_referencias_se_reintenta(store):
    chain, modelo = chain_with([text_reply('{"respuesta": "se corta'), reply()])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias
    assert modelo.calls == 2


async def test_texto_que_no_es_json_se_reintenta(store):
    chain, modelo = chain_with([text_reply("Claro, te cuento que..."), reply()])
    await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 2


async def test_campo_faltante_falla_con_mensaje_en_castellano(store):
    chain, modelo = chain_with([text_reply('{"referencias": []}')] * 3)
    with pytest.raises(InvalidOutputError, match="falta el campo respuesta"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 3


async def test_referencia_inventada_se_reintenta_y_se_recupera(store):
    chain, modelo = chain_with([reply(fuente="inventado.md"), reply(fuente="chroma.md")])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias[0].fuente == "chroma.md"
    assert modelo.calls == 2


async def test_referencia_inventada_siempre_falla_nombrando_las_fuentes(store):
    chain, _ = chain_with([reply(fuente="inventado.md")] * 3)
    with pytest.raises(InvalidOutputError, match="inventado.md no están entre las fuentes recuperadas"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)


async def test_respuesta_sin_fuentes_se_rechaza(store):
    chain, _ = chain_with([reply(referencias=[])] * 3)
    with pytest.raises(InvalidOutputError, match="no cita ninguna fuente"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)


async def test_no_lo_se_con_referencias_se_rechaza(store):
    chain, _ = chain_with([reply(respuesta="No lo sé")] * 3)
    with pytest.raises(InvalidOutputError, match="'No lo sé' no puede traer referencias"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)


async def test_no_lo_se_valido_pasa(store):
    chain, modelo = chain_with([no_lo_se()])
    respuesta = await get_rag_response("¿Cuál es la capital de Francia?", store, chain, top_k=3)
    assert respuesta.sin_respuesta
    assert modelo.calls == 1


async def test_respuesta_cortada_por_tokens_se_reintenta(store):
    chain, modelo = chain_with([reply(stop_reason="max_tokens"), reply()])
    await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 2


async def test_respuesta_cortada_siempre_falla_como_incompleta(store):
    chain, _ = chain_with([reply(stop_reason="length")] * 3)
    with pytest.raises(IncompleteOutputError, match="falta de tokens"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)


async def test_un_error_que_no_es_de_validacion_no_se_reintenta(store):
    chain, modelo = chain_with([reply()], cls=FailingChat)
    with pytest.raises(ConnectionError, match="sin red"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 1


async def test_con_cero_reintentos_hay_un_solo_intento(store):
    chain, modelo = chain_with([reply(referencias=[])] * 3, retries=0)
    with pytest.raises(InvalidOutputError):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 1


async def test_pregunta_vacia_o_demasiado_larga_se_rechaza_sin_llamar_al_modelo(store):
    chain, modelo = chain_with([reply()])
    with pytest.raises(ValueError, match="no puede estar vacía"):
        await get_rag_response("   ", store, chain)
    with pytest.raises(ValueError, match=f"demasiado larga: tiene {MAX_QUERY_LENGTH + 1} caracteres"):
        await get_rag_response("x" * (MAX_QUERY_LENGTH + 1), store, chain)
    assert modelo.calls == 0


async def test_pregunta_en_el_limite_se_procesa(store):
    chain, modelo = chain_with([no_lo_se()])
    await get_rag_response("x" * MAX_QUERY_LENGTH, store, chain, top_k=3)
    assert modelo.calls == 1


async def test_los_logs_incluyen_los_tiempos_y_las_fuentes(store, caplog):
    caplog.set_level(logging.INFO, logger="rag")
    chain, _ = chain_with([text_reply("no json"), reply()])
    await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    mensajes = [r.getMessage() for r in caplog.records]
    assert mensajes.count("Llamando al modelo") == 2
    assert len([m for m in mensajes if re.fullmatch(r"Llamada al modelo: \d+ ms", m)]) == 2
    assert any(m.startswith("Fragmentos recuperados: chroma.md#0") for m in mensajes)
    assert any("no validó" in m for m in mensajes)
    assert "Respuesta validada" in mensajes
    tiempos = [m for m in mensajes if m.startswith("Tiempos: recuperación")]
    assert len(tiempos) == 1
    assert "modelo" in tiempos[0] and "total" in tiempos[0] and " ms" in tiempos[0]


async def test_varias_preguntas_en_paralelo_con_la_misma_cadena(store):
    modelo = ContextChat(responses=[])
    chain = build_chain(modelo, 2)
    preguntas = {
        "¿Dónde guarda ChromaDB los vectores?": "chroma.md",
        "¿Qué es un embedding numérico?": "embeddings.txt",
        "¿Cómo funciona la espera exponencial?": "reintentos.md",
        "¿Qué hace la fotosíntesis con la luz?": "plantas.txt",
    }
    respuestas = await asyncio.gather(*(get_rag_response(p, store, chain, top_k=4) for p in preguntas))
    for fuente, respuesta in zip(preguntas.values(), respuestas):
        assert respuesta.referencias[0].fuente == fuente
    assert modelo.calls == len(preguntas)


async def test_el_numero_de_fragmentos_del_contexto_respeta_top_k(store):
    resultados = await retrieve(store, "vectores", 3)
    assert len(resultados) == 3
    modelo = ContextChat(responses=[])
    chain = build_chain(modelo, 2)
    await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=4)
    assert modelo.prompts[-1].count("Fuente: ") == 4


async def test_no_lo_se_con_texto_extra_sin_fuentes_se_rechaza(store):
    chain, modelo = chain_with([reply(respuesta="No lo sé, pero la capital de Francia es París", referencias=[])] * 3)
    with pytest.raises(InvalidOutputError, match="la respuesta no cita ninguna fuente"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 3


async def test_cita_inventada_se_reintenta_y_se_recupera(store):
    invented = reply(referencias=[{"fuente": "chroma.md", "fragmento": "frase que no está"}])
    chain, modelo = chain_with([invented, reply()])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias[0].fragmento == "una carpeta local persistente"
    assert modelo.calls == 2


async def test_cita_inventada_siempre_falla_con_el_fragmento(store):
    invented = reply(referencias=[{"fuente": "chroma.md", "fragmento": "frase que no está"}])
    chain, modelo = chain_with([invented] * 3)
    with pytest.raises(InvalidOutputError, match="la referencia a chroma.md cita un fragmento que no aparece en lo recuperado: 'frase que no está'"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 3


async def test_cita_ignora_mayusculas_tildes_y_puntuacion(store):
    chain, _ = chain_with([reply(referencias=[{"fuente": "chroma.md", "fragmento": "¡ÚNA, CARPETA; LOCAL: PERSISTENTE!"}])])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias


async def test_bloque_markdown_con_json_valido_se_acepta(store):
    chain, modelo = chain_with([text_reply(f"```json\n{reply().content}\n```")])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias
    assert modelo.calls == 1


async def test_texto_irreparable_falla_con_mensaje_claro(store):
    chain, modelo = chain_with([text_reply("texto irreparable")] * 3)
    with pytest.raises(InvalidOutputError, match="no es un JSON válido con el formato pedido"):
        await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert modelo.calls == 3


async def test_json_casi_completo_con_campos_y_citas_reales_se_acepta(store):
    chain, modelo = chain_with([text_reply(reply().content[:-1], stop_reason="end_turn")])
    respuesta = await get_rag_response("¿Dónde guarda ChromaDB los vectores?", store, chain, top_k=3)
    assert respuesta.referencias[0].fragmento == "una carpeta local persistente"
    assert modelo.calls == 1


async def test_cuatro_llamadas_al_modelo_se_solapan(store):
    barrier = threading.Barrier(4, timeout=5)

    class BarrierChat(FakeChat):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            barrier.wait()
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    chain, modelo = chain_with([reply()], cls=BarrierChat)
    respuestas = await asyncio.wait_for(asyncio.gather(*(
        get_rag_response(f"¿Dónde guarda ChromaDB los vectores? {i}", store, chain, top_k=4)
        for i in range(4)
    )), timeout=10)
    assert len(respuestas) == 4
    assert all(r.referencias[0].fuente == "chroma.md" for r in respuestas)
    assert modelo.calls == 4


def test_get_default_store_inicializa_una_sola_vez_entre_hilos(monkeypatch, store_settings):
    import embeddings
    import ingest
    import rag

    barrier = threading.Barrier(4, timeout=5)
    counter_lock = threading.Lock()
    calls = []
    expected_store = object()

    def build(settings):
        with counter_lock:
            calls.append(settings)
        time.sleep(0.05)
        return object()

    def get_store():
        barrier.wait()
        return rag.get_default_store()

    rag._get_default_store.cache_clear()
    rag._get_default_settings.cache_clear()
    monkeypatch.setattr(rag, "load_store_settings", lambda: store_settings)
    monkeypatch.setattr(embeddings, "build_embeddings", build)
    monkeypatch.setattr(ingest, "open_store", lambda settings, model: expected_store)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            stores = list(pool.map(lambda _: get_store(), range(4)))
        assert all(s is expected_store for s in stores)
        assert calls == [store_settings]
    finally:
        rag._get_default_store.cache_clear()
        rag._get_default_settings.cache_clear()
