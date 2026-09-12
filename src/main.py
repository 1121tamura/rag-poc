from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.infrastructure.history.sqlite_conversation_repository import init_db
from src.interfaces.api.conversation_router import router as conversation_router
from src.interfaces.api.query_router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # 会話履歴テーブルの作成（起動時に1回だけ）
    yield


app = FastAPI(title="RAG Support System", lifespan=lifespan)
app.include_router(router)
app.include_router(conversation_router)
