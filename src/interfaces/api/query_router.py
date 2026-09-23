import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.application.rag_service import generate_answer, save_exchange
from src.infrastructure.history import sqlite_conversation_repository as repository
from src.interfaces.api.dependencies import get_current_user_id

logger = logging.getLogger(__name__)

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
        t0 = time.perf_counter()
        result = await generate_answer(
            request.query,
            document_type=request.document_type,
            conversation_id=request.conversation_id,
            user_id=user_id,
        )
        t1 = time.perf_counter()  # 検索・書き換えの完了

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
        first_token_time: float | None = None
        async for chunk in result.response_stream:
            if chunk.delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()  # 最初のトークンが来た瞬間（prefill完了の目安）
                answer_parts.append(chunk.delta)
                yield json.dumps({"type": "token", "content": chunk.delta}, ensure_ascii=False) + "\n"
        t2 = time.perf_counter()  # 生成完了

        answer_text = "".join(answer_parts)
        logger.info(
            "timing: search+rewrite=%.2fs, prefill(初動)=%.2fs, decode(生成)=%.2fs, output_chars=%d",
            t1 - t0,
            (first_token_time - t1) if first_token_time is not None else -1,
            (t2 - first_token_time) if first_token_time is not None else -1,
            len(answer_text),
        )

        # 保存は回答が出そろってから。質問を先に保存すると、その質問自身が履歴として読み出される
        title = None
        if request.conversation_id is not None:
            title = await save_exchange(
                request.conversation_id,
                request.query,
                answer_text,
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
