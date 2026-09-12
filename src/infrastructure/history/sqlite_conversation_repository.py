import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
"""


def init_db() -> None:
    """テーブル・インデックスを作成する。アプリ起動時に1回だけ呼ぶ"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executescript(SCHEMA)


def get_connection() -> sqlite3.Connection:
    """SQLite接続を返す"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # 接続ごとの設定。ON DELETE CASCADEを有効化
    return conn


@contextmanager
def _connect():
    """接続のクローズとトランザクション制御をまとめて行う"""
    with closing(get_connection()) as conn:
        with conn:  # 正常終了でCOMMIT、例外でROLLBACK
            yield conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(user_id: str, title: str) -> str:
    """会話を新規作成し、生成したIDを返す"""
    conversation_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user_id, title, now, now),
        )
    return conversation_id


def list_conversations(user_id: str) -> list[dict]:
    """利用者の会話一覧を更新日時の降順で返す"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_message(conversation_id: str, role: str, content: str, sources: list[dict] | None = None) -> str:
    """メッセージを追加し、所属する会話のupdated_atも更新する"""
    message_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_id,
                conversation_id,
                role,
                content,
                json.dumps(sources, ensure_ascii=False) if sources else None,
                now,
            ),
        )
        # 一覧の並び順（updated_at降順）に反映させるため同じトランザクションで更新する
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return message_id


def owns_conversation(conversation_id: str, user_id: str) -> bool:
    """会話が指定した利用者のものかを返す。保存前の所有者チェックに使う"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
        ).fetchone()
    return row is not None


def update_title(conversation_id: str, title: str) -> None:
    """会話タイトルを更新する。作成時の仮タイトルを初回質問時にLLM生成の値へ差し替えるために使う"""
    with _connect() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def get_messages(conversation_id: str, user_id: str) -> list[dict]:
    """会話のメッセージを時系列で返す。他人の会話は取得できない"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT m.id, m.role, m.content, m.sources, m.created_at "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND c.user_id = ? "
            "ORDER BY m.created_at",
            (conversation_id, user_id),
        ).fetchall()
    return [{**dict(row), "sources": json.loads(row["sources"]) if row["sources"] else None} for row in rows]


def get_recent_messages(conversation_id: str, user_id: str, limit: int) -> list[dict]:
    """プロンプトに含める直近のメッセージを時系列で返す（sourcesは不要なので含めない）"""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT m.role, m.content "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND c.user_id = ? "
            "ORDER BY m.created_at DESC LIMIT ?",
            (conversation_id, user_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """会話を削除する。所有者本人のみ削除でき、削除できたかどうかを返す"""
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        deleted = cursor.rowcount > 0
    return deleted
