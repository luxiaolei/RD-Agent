from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    log_llm_chat_content: bool = True

    max_retry: int = 10
    retry_wait_seconds: int = 1
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10

    # Behavior of returning answers to the same question when caching is enabled
    use_auto_chat_cache_seed_gen: bool = False
    init_chat_cache_seed: int = 42

    # Chat configs
    chat_model: str = "gemini-1.5-flash-latest"
    chat_max_tokens: int = 120000
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = 120000
    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."

    # Embedding configs
    embedding_model: str = "text-embedding-ada-002"
    embedding_max_str_num: int = 50

    # ProxyLLM configs


LLM_SETTINGS = LLMSettings()
