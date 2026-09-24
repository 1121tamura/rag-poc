import os

from llama_index.llms.ollama import Ollama

MODEL_NAME = "qwen3:8b"
REQUEST_TIMEOUT = 300.0
# Ollamaの既定コンテキスト長は4,096でモデル本来の40,960より大幅に小さく、
# 超過分は警告なく捨てられるため明示指定する（仕様書7.4）
CONTEXT_WINDOW = 8192


def get_llm() -> Ollama:
    """OllamaのQwen3 8Bモデルへの接続を返す。思考モードはモデル既定のON（参照文書をまたぐ推論のため）"""
    return Ollama(
        model=MODEL_NAME,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        request_timeout=REQUEST_TIMEOUT,
        context_window=CONTEXT_WINDOW,
    )


def get_llm_without_thinking() -> Ollama:
    """補助タスク（質問の言い換え・タイトル生成）用の接続を返す。

    Qwen3は思考モードが既定でONで、回答本文より長い思考トークンを生成することがある
    （実測で本文1502文字に対し思考2531文字）。言い換えやタイトル生成に推論は要らないため、
    ここだけ思考モードを切って応答を速くする。回答生成側はONのまま残す。
    """
    return Ollama(
        model=MODEL_NAME,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        request_timeout=REQUEST_TIMEOUT,
        context_window=CONTEXT_WINDOW,
        thinking=False,
    )
