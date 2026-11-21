from pathlib import Path

import numpy as np
import pytest

import embeddings
from config import load_store_settings
from ingest import build_splitter, load_documents, split_documents

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / ".cache" / "modelos"
pytestmark = pytest.mark.skipif(not CACHE_DIR.is_dir(), reason="El modelo local no está en .cache/modelos")


@pytest.fixture(scope="module")
def real_embeddings(offline_tests):
    text_embedding = embeddings.TextEmbedding
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(embeddings, "TextEmbedding", lambda **kwargs: text_embedding(**kwargs, local_files_only=True, threads=2))
        yield embeddings.build_embeddings(load_store_settings({}))


def test_fragmentos_reales_no_superan_el_limite_del_modelo(real_embeddings):
    fragmentos = split_documents(
        load_documents(BASE_DIR / "data"), build_splitter(500, 50, real_embeddings.count_tokens)
    )
    assert fragmentos
    assert all(real_embeddings.count_tokens(f.page_content) + 2 <= real_embeddings.max_tokens for f in fragmentos)


def test_count_tokens_no_trunca_textos_largos(real_embeddings):
    texto = "texto " * 700
    assert real_embeddings.max_tokens == 512
    assert real_embeddings.count_tokens(texto) > 512
    assert len(real_embeddings.model.model.tokenizer.encode(texto).ids) == 512


def test_textos_distintos_despues_del_token_512_dan_vectores_iguales(real_embeddings):
    prefix = "texto " * 600
    textos = [prefix + "capital de Francia", prefix + "plantas y fotosíntesis"]
    assert real_embeddings.count_tokens(prefix) > 512
    assert real_embeddings.tokenizer.encode(textos[0]).ids != real_embeddings.tokenizer.encode(textos[1]).ids
    vectores = real_embeddings.embed_documents(textos)
    np.testing.assert_array_equal(vectores[0], vectores[1])
