import os
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 実行環境に応じて切り替える。GPU環境では "cuda"、CPU環境では "cpu"（既定）。
# GPUで動かすにはCUDA版のPyTorchが必要（pyproject.tomlの[tool.uv.sources]を参照）
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
USE_FP16 = EMBEDDING_DEVICE.startswith("cuda")  # fp16はGPUでのみ有効


@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    """BGE-M3のdense embeddingモデルを返す。モデルが重いためプロセス内で1度だけ読み込む"""
    return HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        cache_folder=os.environ.get("HF_HOME"),
        device=EMBEDDING_DEVICE,
    )


@lru_cache(maxsize=1)
def _get_sparse_model() -> BGEM3FlagModel:
    """sparse embedding用のBGE-M3モデルを返す。プロセス内で1度だけ読み込む"""
    return BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=USE_FP16,
        devices=EMBEDDING_DEVICE,
        cache_dir=os.environ.get("HF_HOME"),
    )


def sparse_doc_vectors(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    """BGE-M3のsparse embeddingを生成する。Qdrantのsparse_doc_fnとして使用する"""
    output = _get_sparse_model().encode_corpus(
        texts, return_dense=False, return_sparse=True, return_colbert_vecs=False
    )
    indices_list = [[int(token_id) for token_id in weights] for weights in output["lexical_weights"]]
    values_list = [[float(weight) for weight in weights.values()] for weights in output["lexical_weights"]]
    return indices_list, values_list
