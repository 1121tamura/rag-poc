from fastapi import APIRouter, Depends, HTTPException

from src.infrastructure.history import sqlite_conversation_repository as repository
from src.interfaces.api.dependencies import get_current_user_id

router = APIRouter(prefix="/conversations")

PLACEHOLDER_TITLE = "新しい会話"  # 初回質問時にLLM生成のタイトルへ差し替える（仕様書7.4）


@router.get("")
def list_conversations(user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """利用者の会話一覧を updated_at 降順で返す"""
    return repository.list_conversations(user_id)


@router.post("", status_code=201)
def create_conversation(user_id: str = Depends(get_current_user_id)) -> dict:
    """会話を新規作成する。タイトルは初回質問時に差し替えるため仮の値を入れる"""
    conversation_id = repository.create_conversation(user_id, PLACEHOLDER_TITLE)
    return {"id": conversation_id, "title": PLACEHOLDER_TITLE}


@router.get("/{conversation_id}")
def get_messages(conversation_id: str, user_id: str = Depends(get_current_user_id)) -> list[dict]:
    """会話のメッセージ一覧を時系列で返す"""
    return repository.get_messages(conversation_id, user_id)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user_id: str = Depends(get_current_user_id)) -> None:
    """会話を削除する。所有者本人のみ削除できる"""
    if not repository.delete_conversation(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="会話が見つかりません")
