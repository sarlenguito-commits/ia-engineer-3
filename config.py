import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator

from models import BUILDERS, MODEL_PREFIXES
from schemas import explicar_errores

BASE_DIR = Path(__file__).parent
MIN_CHUNK_SIZE = 500


class ConfigError(Exception):
    pass


class StoreSettings(BaseModel):
    data_dir: Path
    vectorstore_dir: Path
    embedding_cache_dir: Path
    collection: str = Field(default="documentos", min_length=1)
    embedding_model: str = Field(default="jinaai/jina-embeddings-v2-base-es", min_length=1)
    chunk_size: int = Field(default=500, ge=MIN_CHUNK_SIZE, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=4, ge=3, le=5)

    @model_validator(mode="after")
    def validar_solapamiento(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap tiene que ser menor que chunk_size; se recibió {self.chunk_overlap} con chunk_size {self.chunk_size}"
            )
        return self


class LLMSettings(BaseModel):
    provider: str
    model: str = Field(min_length=1)
    api_key: SecretStr
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    timeout: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validar_coherencia(self):
        if self.provider not in BUILDERS:
            raise ValueError(f"proveedor desconocido: {self.provider}. Opciones: {', '.join(BUILDERS)}")
        prefijos = MODEL_PREFIXES[self.provider]
        if not self.model.lower().startswith(prefijos):
            raise ValueError(
                f"el modelo {self.model!r} no parece de {self.provider}; tendría que empezar con {' o '.join(prefijos)}"
            )
        return self


def read_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is None:
        load_dotenv(BASE_DIR / ".env")
        return os.environ
    return env


def resolve_path(valor: str | None, por_defecto: str) -> Path:
    ruta = Path(valor.strip() if valor and valor.strip() else por_defecto)
    return ruta if ruta.is_absolute() else BASE_DIR / ruta


def load_store_settings(env: Mapping[str, str] | None = None) -> StoreSettings:
    env = read_env(env)
    opcionales = {
        "collection": env.get("RAG_COLLECTION"),
        "embedding_model": env.get("EMBEDDING_MODEL"),
        "chunk_size": env.get("CHUNK_SIZE"),
        "chunk_overlap": env.get("CHUNK_OVERLAP"),
        "top_k": env.get("TOP_K"),
    }
    try:
        return StoreSettings(
            data_dir=resolve_path(env.get("RAG_DATA_DIR"), "data"),
            vectorstore_dir=resolve_path(env.get("RAG_VECTORSTORE_DIR"), "vectorstore"),
            embedding_cache_dir=resolve_path(env.get("EMBEDDING_CACHE_DIR"), ".cache/modelos"),
            **{k: v for k, v in opcionales.items() if v},
        )
    except ValidationError as error:
        raise ConfigError(f"Configuración inválida: {explicar_errores(error)}") from error


def load_llm_settings(env: Mapping[str, str] | None = None) -> LLMSettings:
    env = read_env(env)
    provider = env.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in BUILDERS:
        raise ConfigError(f"Proveedor desconocido: {provider}. Opciones: {', '.join(BUILDERS)}")
    prefix = provider.upper()
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(f"Falta {prefix}_API_KEY en el .env")
    model = env.get(f"{prefix}_MODEL", "").strip()
    if not model:
        raise ConfigError(f"Falta {prefix}_MODEL en el .env")
    opcionales = {
        "max_tokens": env.get("LLM_MAX_TOKENS"),
        "timeout": env.get("LLM_TIMEOUT"),
        "max_retries": env.get("LLM_MAX_RETRIES"),
    }
    try:
        return LLMSettings(
            provider=provider,
            model=model,
            api_key=SecretStr(api_key),
            **{k: v for k, v in opcionales.items() if v},
        )
    except ValidationError as error:
        raise ConfigError(f"Configuración inválida: {explicar_errores(error)}") from error
