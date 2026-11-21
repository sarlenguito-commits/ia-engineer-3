import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

from config import StoreSettings


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        tokenizer = self.model.model.tokenizer
        self.max_tokens: int = tokenizer.truncation["max_length"]
        self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self.tokenizer.no_truncation()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def build_embeddings(settings: StoreSettings) -> LocalEmbeddings:
    return LocalEmbeddings(settings.embedding_model, settings.embedding_cache_dir)
