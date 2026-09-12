import os

from llama_index.llms.ollama import Ollama

MODEL_NAME = "qwen3:8b"
REQUEST_TIMEOUT = 300.0
# Ollamaの既定コンテキスト長は4,096でモデル本来の40,960より大幅に小さく、
# 超過分は警告なく捨てられるため明示指定する（仕様書7.4）
CONTEXT_WINDOW = 8192


def get_llm() -> Ollama:
    """OllamaのQwen3 8Bモデルへの接続を返す"""
    return Ollama(
        model=MODEL_NAME,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        request_timeout=REQUEST_TIMEOUT,
        context_window=CONTEXT_WINDOW,
    )
