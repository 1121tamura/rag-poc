import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.infrastructure.ingestion.loaders.markdown_loader import attach_metadata, load_markdown
from src.infrastructure.llama_index_factory import get_embed_model
from src.infrastructure.qdrant_vector_store import COLLECTION_NAME, get_vector_store

# 本番ではコードをイメージに焼き込むため、状態ファイルはボリューム側に置けるようにする。
# 既定は開発時と同じプロジェクトルート直下（仕様書6.2）
STATE_FILE = Path(os.environ.get("INGESTION_STATE_FILE", ".ingestion_state.json"))


def _load_last_run_time(state_file: Path = STATE_FILE) -> datetime | None:
    """前回Ingestion実行時刻を状態ファイルから読み込む。初回実行時はNoneを返す"""
    if not state_file.exists():
        return None
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return datetime.fromisoformat(data["last_run"])


def _iter_changed_files(docs_dir: Path, last_run_time: datetime | None) -> Iterator[Path]:
    """docs_dir配下のMarkdownファイルのうち、前回実行時刻より更新されたものを列挙する"""
    for path in docs_dir.rglob("*.md"):
        if last_run_time is None or datetime.fromtimestamp(path.stat().st_mtime) > last_run_time:
            yield path


def _delete_existing_chunks(client: QdrantClient, collection_name: str, file_path: Path) -> None:
    """指定ファイルに対応する既存チャンクを、file_pathキーでフィルタしてQdrantから削除する"""
    if not client.collection_exists(collection_name):
        return
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="file_path", match=models.MatchValue(value=str(file_path)))]
            )
        ),
    )


def _save_last_run_time(run_time: datetime, state_file: Path = STATE_FILE) -> None:
    """現在の実行時刻を状態ファイルに保存する"""
    state_file.write_text(json.dumps({"last_run": run_time.isoformat()}), encoding="utf-8")


def run(docs_dir: Path) -> None:
    """docs_dir配下の変更されたMarkdownファイルをQdrantに取り込む"""
    last_run_time = _load_last_run_time()
    run_time = datetime.now()

    embed_model = get_embed_model()
    vector_store = get_vector_store()

    for file_path in _iter_changed_files(docs_dir, last_run_time):
        document_type = file_path.relative_to(docs_dir).parts[0]
        nodes = load_markdown(file_path)
        nodes = attach_metadata(
            nodes,
            document_type=document_type,
            file_path=file_path,
            file_format="md",
            updated_at=datetime.fromtimestamp(file_path.stat().st_mtime),
        )
        for node in nodes:
            node.embedding = embed_model.get_text_embedding(node.text)

        _delete_existing_chunks(vector_store.client, COLLECTION_NAME, file_path)
        vector_store.add(nodes)

    _save_last_run_time(run_time)
