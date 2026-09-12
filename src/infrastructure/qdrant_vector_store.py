import os

import qdrant_client
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import relative_score_fusion

from src.infrastructure.llama_index_factory import sparse_doc_vectors

COLLECTION_NAME = "documents"
DENSE_SCORE_KEY = "_dense_score"
SPARSE_SCORE_KEY = "_sparse_score"


def hybrid_fusion_with_raw_scores(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 2,
) -> VectorStoreQueryResult:
    """既定のrelative_score_fusionに委譲しつつ、融合前の生スコアをノードのメタデータに残す。
    融合後スコアは結果セット内で正規化された相対値であり関連度の絶対値ではないため、
    閾値判定（仕様書7.3対策3）には融合前のスコアが必要になる"""
    dense_scores = dict(zip((n.node_id for n in dense_result.nodes or []), dense_result.similarities or []))
    sparse_scores = dict(zip((n.node_id for n in sparse_result.nodes or []), sparse_result.similarities or []))

    fused = relative_score_fusion(dense_result, sparse_result, alpha=alpha, top_k=top_k)
    for node in fused.nodes or []:
        node.metadata[DENSE_SCORE_KEY] = dense_scores.get(node.node_id)
        node.metadata[SPARSE_SCORE_KEY] = sparse_scores.get(node.node_id)
    return fused


def get_vector_store() -> QdrantVectorStore:
    """Qdrantベクトルストアへの接続を返す（ハイブリッド検索対応）"""
    client = qdrant_client.QdrantClient(
        host=os.environ.get("QDRANT_HOST", "qdrant"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
    )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        sparse_doc_fn=sparse_doc_vectors,
        sparse_query_fn=sparse_doc_vectors,
        hybrid_fusion_fn=hybrid_fusion_with_raw_scores,
    )
