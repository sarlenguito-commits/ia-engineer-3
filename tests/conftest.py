import socket

import pytest
from fakes import DOCUMENTOS, FakeEmbeddings, words

from config import StoreSettings
from ingest import ingest


@pytest.fixture(scope="session", autouse=True)
def offline_tests():
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def deny_network(*args, **kwargs):
        raise AssertionError("Los tests no pueden acceder a la red")

    def local_connect(connection, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return connect(connection, address)
        deny_network()

    def local_connect_ex(connection, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return connect_ex(connection, address)
        deny_network()

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
        patch.setenv("ANONYMIZED_TELEMETRY", "False")
        patch.setattr(socket.socket, "connect", local_connect)
        patch.setattr(socket.socket, "connect_ex", local_connect_ex)
        patch.setattr(socket, "getaddrinfo", deny_network)
        patch.setattr("config.load_dotenv", lambda *args, **kwargs: None)
        yield


@pytest.fixture
def store_settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for nombre, texto in DOCUMENTOS.items():
        (data / nombre).write_text(texto, encoding="utf-8")
    return StoreSettings(
        data_dir=data,
        vectorstore_dir=tmp_path / "vectorstore",
        embedding_cache_dir=tmp_path / "cache",
        top_k=3,
    )


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def store(store_settings, embeddings):
    return ingest(store_settings, embeddings, length_function=words)
