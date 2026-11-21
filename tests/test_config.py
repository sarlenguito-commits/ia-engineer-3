import pytest

from config import BASE_DIR, ConfigError, load_llm_settings, load_store_settings

LLM_ENV = {"ANTHROPIC_API_KEY": "clave-de-prueba", "ANTHROPIC_MODEL": "claude-haiku-5-5"}


def test_configuracion_de_la_base_por_defecto():
    settings = load_store_settings({})
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.top_k == 4
    assert settings.data_dir == BASE_DIR / "data"
    assert settings.vectorstore_dir == BASE_DIR / "vectorstore"
    assert settings.embedding_model == "jinaai/jina-embeddings-v2-base-es"


def test_las_rutas_relativas_salen_de_la_carpeta_del_proyecto():
    settings = load_store_settings({"RAG_DATA_DIR": "docs/notas"})
    assert settings.data_dir == BASE_DIR / "docs" / "notas"


@pytest.mark.parametrize("top_k", ["2", "6", "0"])
def test_top_k_fuera_de_rango(top_k):
    with pytest.raises(ConfigError, match=r"top_k tiene que ser (mayor o igual a 3|menor o igual a 5)"):
        load_store_settings({"TOP_K": top_k})


@pytest.mark.parametrize("top_k", ["3", "5"])
def test_top_k_en_los_limites(top_k):
    assert load_store_settings({"TOP_K": top_k}).top_k == int(top_k)


def test_fragmentos_de_menos_de_500_tokens_se_rechazan():
    with pytest.raises(ConfigError, match="chunk_size tiene que ser mayor o igual a 500; se recibió '300'"):
        load_store_settings({"CHUNK_SIZE": "300"})


def test_chunk_size_justo_en_los_limites():
    with pytest.raises(ConfigError, match="mayor o igual a 500; se recibió '499'"):
        load_store_settings({"CHUNK_SIZE": "499"})
    assert load_store_settings({"CHUNK_SIZE": "500"}).chunk_size == 500
    assert load_store_settings({"CHUNK_SIZE": "2000"}).chunk_size == 2000
    with pytest.raises(ConfigError, match="menor o igual a 2000; se recibió '2001'"):
        load_store_settings({"CHUNK_SIZE": "2001"})


def test_chunk_overlap_justo_en_el_limite():
    assert load_store_settings({"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "499"}).chunk_overlap == 499
    with pytest.raises(ConfigError, match="chunk_overlap tiene que ser menor que chunk_size"):
        load_store_settings({"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "500"})
    with pytest.raises(ConfigError, match="chunk_overlap tiene que ser mayor o igual a 0; se recibió '-1'"):
        load_store_settings({"CHUNK_OVERLAP": "-1"})


def test_solapamiento_igual_o_mayor_que_el_fragmento():
    with pytest.raises(ConfigError, match="chunk_overlap tiene que ser menor que chunk_size"):
        load_store_settings({"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "500"})


def test_valor_que_no_es_numero():
    with pytest.raises(ConfigError, match="top_k tiene que ser un número; se recibió 'muchos'"):
        load_store_settings({"TOP_K": "muchos"})


def test_configuracion_del_modelo_valida():
    settings = load_llm_settings(LLM_ENV)
    assert settings.provider == "anthropic"
    assert settings.max_tokens == 1024
    assert "clave-de-prueba" not in repr(settings)


def test_proveedor_desconocido():
    with pytest.raises(ConfigError, match="Proveedor desconocido: gemini"):
        load_llm_settings({**LLM_ENV, "LLM_PROVIDER": "gemini"})


def test_falta_la_clave_o_el_modelo():
    with pytest.raises(ConfigError, match="Falta ANTHROPIC_API_KEY"):
        load_llm_settings({"ANTHROPIC_MODEL": "claude-haiku-5-5"})
    with pytest.raises(ConfigError, match="Falta ANTHROPIC_MODEL"):
        load_llm_settings({"ANTHROPIC_API_KEY": "x"})


def test_modelo_que_no_corresponde_al_proveedor():
    env = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "x", "OPENAI_MODEL": "claude-haiku-5-5"}
    with pytest.raises(ConfigError, match="no parece de openai"):
        load_llm_settings(env)


def test_max_tokens_y_reintentos_fuera_de_rango():
    with pytest.raises(ConfigError, match="max_tokens tiene que ser mayor o igual a 64"):
        load_llm_settings({**LLM_ENV, "LLM_MAX_TOKENS": "10"})
    with pytest.raises(ConfigError, match="max_retries tiene que ser menor o igual a 5"):
        load_llm_settings({**LLM_ENV, "LLM_MAX_RETRIES": "9"})
