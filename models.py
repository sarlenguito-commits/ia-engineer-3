from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

BUILDERS: dict[str, Callable[..., BaseChatModel]] = {}
MODEL_PREFIXES: dict[str, tuple[str, ...]] = {}


def register_provider(name: str, prefixes: tuple[str, ...]):
    def decorator(builder: Callable[..., BaseChatModel]):
        BUILDERS[name] = builder
        MODEL_PREFIXES[name] = prefixes
        return builder

    return decorator


@register_provider("anthropic", ("claude",))
def build_anthropic(model: str, api_key: str, max_tokens: int, timeout: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, api_key=api_key, max_tokens=max_tokens, timeout=timeout, max_retries=2)


@register_provider("openai", ("gpt", "o1", "o3", "o4", "chatgpt"))
def build_openai(model: str, api_key: str, max_tokens: int, timeout: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=api_key, max_tokens=max_tokens, timeout=timeout, max_retries=2)


def create_model(provider: str, model: str, api_key: str, max_tokens: int, timeout: float) -> BaseChatModel:
    return BUILDERS[provider](model, api_key, max_tokens, timeout)
