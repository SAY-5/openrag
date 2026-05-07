from services.inference.reranker_clients import InfinityReranker as InfinityRerankerAdapter  # noqa: F401
from services.inference.reranker_clients import OpenAIReranker as OpenAIRerankerAdapter  # noqa: F401

from .base import BaseReranker


class RerankerFactory:
    @staticmethod
    def get_reranker(config) -> BaseReranker:
        provider = config.reranker.provider
        if provider == "infinity":
            from .infinity import InfinityReranker

            return InfinityReranker(config)
        elif provider == "openai":
            from .openai import OpenAIReranker

            return OpenAIReranker(config)
        else:
            raise ValueError(f"Unsupported reranker provider: {provider}")
