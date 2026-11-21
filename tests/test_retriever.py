import pytest

from retriever import EmptyStoreError, retrieve


async def test_devuelve_la_fuente_mas_parecida_primero(store):
    resultados = await retrieve(store, "¿Dónde guarda los vectores ChromaDB en una carpeta local?", 3)
    fuentes = [documento.metadata["fuente"] for documento, _ in resultados]
    assert fuentes[0] == "chroma.md"
    similitudes = [similitud for _, similitud in resultados]
    assert similitudes == sorted(similitudes, reverse=True)


async def test_respeta_top_k(store):
    assert len(await retrieve(store, "embedding vector numérico", 3)) == 3
    assert len(await retrieve(store, "embedding vector numérico", 1)) == 1


async def test_la_similitud_es_mayor_para_lo_relacionado(store):
    relacionado = await retrieve(store, "qué es un embedding vector numérico", 1)
    sin_relacion = await retrieve(store, "capital de Francia turismo París", 1)
    assert relacionado[0][1] > sin_relacion[0][1]


async def test_base_vacia(store_settings, embeddings):
    from ingest import open_store

    vacia = open_store(store_settings, embeddings)
    with pytest.raises(EmptyStoreError, match="corré ingest.py primero"):
        await retrieve(vacia, "cualquier cosa", 3)
