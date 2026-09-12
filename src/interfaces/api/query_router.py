import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.application.rag_service import generate_answer, save_exchange
from src.infrastructure.history import sqlite_conversation_repository as repository
from src.interfaces.api.dependencies import get_current_user_id

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    document_type: str | None = None
    conversation_id: str | None = None


@router.post("/query")
async def query(request: QueryRequest, user_id: str = Depends(get_current_user_id)) -> StreamingResponse:
    """NDJSON形式でsources→token→doneの順にストリーミング返却する（仕様書7.2・7.4）"""
    # 他人の会話に書き込めないよう、生成に入る前に所有者を確認する
    if request.conversation_id is not None and not repository.owns_conversation(request.conversation_id, user_id):
        raise HTTPException(status_code=404, detail="会話が見つかりません")

    async def event_stream():
        result = await generate_answer(
            request.query,
            document_type=request.document_type,
            conversation_id=request.conversation_id,
            user_id=user_id,
        )

        # ソースは検索完了時点（LLM生成前）で確定済みなので、生成を待たず先頭行で返す
        sources = [
            {
                "file_path": n.node.metadata.get("file_path"),
                "chapter_title": n.node.metadata.get("chapter_title"),
                "score": n.score,
            }
            for n in result.nodes
        ]
        yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"

        # chunk.deltaは今回分の差分のみ（累積全文ではない）。生成終盤など
        # 新規テキストが無いチャンクもあるため空delta行は送らない
        answer_parts: list[str] = []
        async for chunk in result.response_stream:
            if chunk.delta:
                answer_parts.append(chunk.delta)
                yield json.dumps({"type": "token", "content": chunk.delta}, ensure_ascii=False) + "\n"

        # 保存は回答が出そろってから。質問を先に保存すると、その質問自身が履歴として読み出される
        title = None
        if request.conversation_id is not None:
            title = await save_exchange(
                request.conversation_id,
                request.query,
                "".join(answer_parts),
                sources,
                result.is_first_question,
            )

        yield json.dumps(
            {
                "type": "done",
                "is_history_trimmed": result.is_history_trimmed,
                "is_query_rewritten": result.is_query_rewritten,
                "title": title,
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
