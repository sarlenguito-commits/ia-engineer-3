import chromadb
import pytest
from fakes import DOCUMENTOS, FakeEmbeddings, words

from ingest import (
    EmbeddingMismatchError,
    IngestError,
    build_splitter,
    ingest,
    load_documents,
    open_store,
    split_documents,
)


def test_lee_solo_txt_y_md_en_orden(tmp_path):
    (tmp_path / "b.md").write_text("segundo", encoding="utf-8")
    (tmp_path / "a.txt").write_text("primero", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("ignorado", encoding="utf-8")
    (tmp_path / "vacio.txt").write_text("   ", encoding="utf-8")
    documentos = load_documents(tmp_path)
    assert [d.metadata["fuente"] for d in documentos] == ["a.txt", "b.md"]


def test_carpeta_inexistente_o_sin_documentos(tmp_path):
    with pytest.raises(IngestError, match="No existe la carpeta de documentos"):
        load_documents(tmp_path / "no-existe")
    with pytest.raises(IngestError, match="No hay archivos .txt ni .md"):
        load_documents(tmp_path)
    (tmp_path / "a.txt").write_text("  ", encoding="utf-8")
    with pytest.raises(IngestError, match="están vacíos"):
        load_documents(tmp_path)


def test_los_fragmentos_respetan_el_tamano_y_el_solapamiento():
    texto = " ".join(f"palabra{i}" for i in range(1200))
    splitter = build_splitter(500, 50, length_function=words)
    partes = splitter.split_text(texto)
    assert len(partes) == 3
    assert all(words(p) <= 500 for p in partes)
    assert partes[0].split()[-50:] == partes[1].split()[:50]


def test_el_splitter_usa_la_funcion_de_conteo_por_palabras():
    splitter = build_splitter(500, 50, words)
    partes = splitter.split_text(" ".join(["Una oración de prueba sobre bases vectoriales."] * 400))
    assert len(partes) > 1
    assert all(words(p) <= 500 for p in partes)
    assert any(len(p) > 500 for p in partes)


def test_cada_fragmento_conserva_fuente_y_posicion():
    texto = " ".join(f"palabra{i}" for i in range(1200))
    from langchain_core.documents import Document

    fragmentos = split_documents([Document(page_content=texto, metadata={"fuente": "largo.md"})], build_splitter(500, 50, words))
    assert [f.metadata for f in fragmentos] == [{"fuente": "largo.md", "fragmento": i} for i in range(3)]


def test_la_ingesta_indexa_todos_los_archivos(store_settings, embeddings, store):
    assert store.get()["ids"].__len__() == len(DOCUMENTOS)
    assert sorted(store.get()["ids"]) == sorted(f"{nombre}:0" for nombre in DOCUMENTOS)
    assert embeddings.document_calls == 1


def test_la_segunda_ingesta_no_vuelve_a_indexar(store_settings, embeddings, store):
    ingest(store_settings, embeddings, length_function=words)
    ingest(store_settings, embeddings, length_function=words)
    assert embeddings.document_calls == 1


def test_force_reindexa_sin_duplicar(store_settings, embeddings, store):
    nueva = ingest(store_settings, embeddings, force=True, length_function=words)
    assert embeddings.document_calls == 2
    assert len(nueva.get()["ids"]) == len(DOCUMENTOS)


def test_force_en_una_base_nueva_funciona(store_settings, embeddings):
    ingest(store_settings, embeddings, force=True, length_function=words)
    assert embeddings.document_calls == 1


def test_la_base_persiste_entre_sesiones(store_settings, store):
    otra = FakeEmbeddings()
    reabierta = open_store(store_settings, otra)
    assert len(reabierta.get()["ids"]) == len(DOCUMENTOS)
    assert otra.document_calls == 0


def test_modelo_de_embeddings_distinto_se_detecta(store_settings, store):
    with pytest.raises(EmbeddingMismatchError, match="'fake-embeddings' y ahora se quiere usar 'otro-modelo'"):
        open_store(store_settings, FakeEmbeddings("otro-modelo"))
    with pytest.raises(EmbeddingMismatchError):
        ingest(store_settings, FakeEmbeddings("otro-modelo"), length_function=words)


def test_la_coleccion_usa_distancia_coseno_y_guarda_el_modelo(store_settings, store):
    cliente = chromadb.PersistentClient(path=str(store_settings.vectorstore_dir))
    metadata = cliente.get_collection(store_settings.collection).metadata
    assert metadata["hnsw:space"] == "cosine"
    assert metadata["embedding_model"] == "fake-embeddings"


def test_force_con_datos_inexistentes_conserva_la_base(store_settings, embeddings, store):
    anterior = store.get()
    settings = store_settings.model_copy(update={"data_dir": store_settings.data_dir / "no-existe"})
    with pytest.raises(IngestError, match="No existe la carpeta"):
        ingest(settings, embeddings, force=True)
    assert store.get() == anterior


def test_force_con_error_de_embeddings_conserva_la_base(store_settings, store):
    class BrokenEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("No se pueden embeber los textos recibidos")

    anterior = store.get()
    with pytest.raises(RuntimeError, match="No se pueden embeber"):
        ingest(store_settings, BrokenEmbeddings(), force=True)
    assert store.get() == anterior


def test_chunk_size_excesivo_falla_antes_de_tocar_la_coleccion(store_settings, store, monkeypatch):
    anterior = store.get()
    settings = store_settings.model_copy(update={"chunk_size": 511})

    def forbidden_client(*args, **kwargs):
        pytest.fail("No debe abrir la colección con chunk_size inválido")

    with monkeypatch.context() as patch:
        patch.setattr(chromadb, "PersistentClient", forbidden_client)
        with pytest.raises(IngestError, match=r"chunk_size 511.*512 tokens.*usá 510 o menos"):
            ingest(settings, FakeEmbeddings(max_tokens=512), force=True)
    assert store.get() == anterior


def test_chunk_size_en_el_limite_efectivo_pasa(store_settings):
    settings = store_settings.model_copy(update={"chunk_size": 510})
    store = ingest(settings, FakeEmbeddings(max_tokens=512))
    assert len(store.get()["ids"]) == len(DOCUMENTOS)


def test_force_permite_reemplazar_el_modelo_de_embeddings(store_settings, store):
    replacement = ingest(store_settings, FakeEmbeddings("otro-modelo"), force=True)
    assert len(replacement.get()["ids"]) == len(DOCUMENTOS)
    assert replacement._collection.metadata["embedding_model"] == "otro-modelo"
    assert replacement._collection.metadata["hnsw:space"] == "cosine"


def test_sin_contador_exige_length_function(store_settings):
    embeddings = FakeEmbeddings()
    embeddings.count_tokens = None
    with pytest.raises(IngestError, match="Los embeddings no saben contar tokens: pasá length_function"):
        ingest(store_settings, embeddings)
    assert not store_settings.vectorstore_dir.exists()
    assert len(ingest(store_settings, embeddings, length_function=words).get()["ids"]) == len(DOCUMENTOS)


def test_base_existente_se_omite_sin_leer_documentos(store_settings, embeddings, store):
    settings = store_settings.model_copy(update={"data_dir": store_settings.data_dir / "no-existe"})
    reopened = ingest(settings, embeddings)
    assert len(reopened.get()["ids"]) == len(DOCUMENTOS)
    assert embeddings.document_calls == 1
