from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, ChatResponseAsyncGen, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from src.infrastructure.history import sqlite_conversation_repository as repository
from src.infrastructure.llama_index_factory import get_embed_model
from src.infrastructure.ollama_client import get_llm
from src.infrastructure.qdrant_vector_store import DENSE_SCORE_KEY, SPARSE_SCORE_KEY, get_vector_store

TOP_K = 5
PROMPT_TOKEN_BUDGET = 4096  # 仕様書7.4。context_window=8192に対し生成枠とタイムアウトの余裕を残す
GENERATION_RESERVE_TOKENS = 512  # 回答生成のために確保する枠
HISTORY_TURNS = 5  # 仕様書7.4：直近5往復
# 仕様書7.3対策3。融合後スコアではなく融合前の生スコアで判定する。
# 開発用サンプル文書45チャンクでの測定値（関連あり/無関係/挨拶の各質問群の1位スコア）：
#   dense  関連あり 0.6185〜0.7811 ／ 無関係・挨拶 0.4190〜0.4942
#   sparse 関連あり 0.1741〜0.2904 ／ 無関係・挨拶 0.0010〜0.0654
# 両群の中間を採用。実文書の投入後に人手レビューで見直す前提（仕様書9章）
DENSE_RELEVANCE_THRESHOLD = 0.55
SPARSE_RELEVANCE_THRESHOLD = 0.12
SYSTEM_PROMPT = (
    "あなたは購買システムのドキュメントに基づいて質問に回答するアシスタントです。"
    "挨拶や世間話には自然に応答して構いません。"
    "ただし、購買システムに関する質問については、以下の参照文書の情報のみを根拠に回答してください。"
    "参照文書に答えが書かれていない場合は、推測で回答せず、"
    "参照文書のどこまでが分かっていて何が記載されていないのかを具体的に伝えたうえで、"
    "どう質問し直せば見つかりそうかを添えてください。"
)
# 関連度が閾値に満たず参照文書を渡さない場合のプロンプト（仕様書7.3対策3）。
# 挨拶も閾値で弾かれてこちら側に入るため、自然な応答を明記している
NO_DOCUMENT_PROMPT = (
    "あなたは購買システムのドキュメントに基づいて質問に回答するアシスタントです。"
    "今回の質問に関連する社内文書は見つかりませんでした。"
    "挨拶や世間話には自然に応答してください。"
    "購買システムに関する質問の場合は、関連する文書が見つからなかったことを伝えたうえで、"
    "機能名・エラーコード・帳票名など具体的な名称を含めて質問し直すよう案内してください。"
    "推測による回答や、文書に存在しない内容の説明は絶対にしないでください。"
)
REWRITE_PROMPT = (
    "会話の続きとして質問されました。この質問を、単独で意味の通る検索クエリに書き換えてください。\n"
    "履歴を見ても指示語が何を指すか判断できない場合は、推測で補完せず、質問文をそのまま返してください。\n"
    "書き換えた文だけを出力し、説明や引用符は付けないでください。"
)
TITLE_PROMPT = (
    "以下の質問から、会話一覧に表示するタイトルを1つだけ出力してください。\n"
    "20文字以内の日本語の名詞句とし、説明・引用符・句点は付けないでください。"
)
TITLE_MAX_LENGTH = 20


def _estimate_tokens(text: str) -> int:
    """トークン数を安全側（多め）に概算する。実測は約1.34文字/トークンだが1文字＝1トークンとみなす"""
    return len(text)


def _to_turns(history: list[dict]) -> list[list[dict]]:
    """メッセージ列を往復単位にまとめる。回答の保存に失敗した等で対にならない場合も崩れない"""
    turns: list[list[dict]] = []
    for message in history:
        if message["role"] == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim_history(history: list[dict], fixed_prompt_text: str) -> tuple[list[dict], bool]:
    """予算に収まるよう古い往復から履歴を落とす。(残った履歴, 切り詰めが起きたか)を返す"""
    budget = PROMPT_TOKEN_BUDGET - GENERATION_RESERVE_TOKENS - _estimate_tokens(fixed_prompt_text)
    kept: list[dict] = []
    used = 0
    # 新しい往復から順に、予算に収まるものだけ残す（切り詰めは往復単位・仕様書7.4）
    for turn in reversed(_to_turns(history)):
        cost = sum(_estimate_tokens(m["content"]) for m in turn)
        if used + cost > budget:
            break
        kept = turn + kept
        used += cost
    return kept, len(kept) < len(history)


def search(query: str, document_type: str | None = None) -> list[NodeWithScore]:
    """質問文でQdrantをハイブリッド検索し、関連チャンクをスコア付きで返す"""
    index = VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )
    filters = None
    if document_type is not None:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="document_type", value=document_type)])

    retriever = index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=TOP_K,
        filters=filters,
    )
    return retriever.retrieve(query)


def has_relevant_document(nodes: list[NodeWithScore]) -> bool:
    """検索結果に十分な関連度のチャンクがあるかを判定する。
    denseとsparseは得意な検索語が異なるため、どちらかが閾値を超えていれば関連ありとみなす（仕様書8章）"""
    for node_with_score in nodes:
        metadata = node_with_score.node.metadata
        dense = metadata.get(DENSE_SCORE_KEY) or 0.0
        sparse = metadata.get(SPARSE_SCORE_KEY) or 0.0
        if dense >= DENSE_RELEVANCE_THRESHOLD or sparse >= SPARSE_RELEVANCE_THRESHOLD:
            return True
    return False


async def save_exchange(
    conversation_id: str, query: str, answer: str, sources: list[dict], is_first_question: bool
) -> str | None:
    """質問と回答を保存する。初回質問ならタイトルも生成して差し替え、生成したタイトルを返す（仕様書7.4）"""
    repository.add_message(conversation_id, "user", query)
    repository.add_message(conversation_id, "assistant", answer, sources)
    if not is_first_question:
        return None
    title = await generate_title(query)
    repository.update_title(conversation_id, title)
    return title


async def rewrite_query(query: str, history: list[dict]) -> tuple[str, bool]:
    """履歴を踏まえて質問を単独で意味の通る検索クエリに書き換える。(書き換え後のクエリ, 書き換えられたか)を返す"""
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=REWRITE_PROMPT),
        ChatMessage(role=MessageRole.USER, content=f"会話履歴:\n{conversation}\n\n質問: {query}"),
    ]
    response = await get_llm().achat(messages)
    # str(response)は "assistant: " 込みの文字列になるためcontentから取る
    rewritten = (response.message.content or "").strip()
    if not rewritten:
        return query, False  # LLMが空を返した場合は元の質問で検索する
    return rewritten, rewritten != query.strip()


async def generate_title(query: str) -> str:
    """初回質問から会話タイトルを生成する（仕様書7.4）"""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=TITLE_PROMPT),
        ChatMessage(role=MessageRole.USER, content=query),
    ]
    response = await get_llm().achat(messages)
    title = (response.message.content or "").strip()
    # LLMが空や長文を返しても一覧が壊れないよう、字数で切り詰めて質問文にフォールバックする
    return title[:TITLE_MAX_LENGTH] or query[:TITLE_MAX_LENGTH]


@dataclass
class AnswerResult:
    """回答生成の結果。ストリーム本体に加え、UIへ伝える付随情報をまとめる"""

    response_stream: ChatResponseAsyncGen
    nodes: list[NodeWithScore]
    is_history_trimmed: bool  # 履歴の切り詰めが起きたか。UIに通知する（仕様書7.4）
    is_query_rewritten: bool  # 指示語を解決できたか。UI側が案内の要否を判断する（仕様書7.4）
    is_first_question: bool  # 会話の初回質問か。タイトル生成の要否判定に使う（仕様書7.4）
    has_relevant_document: bool  # 関連する参照文書が見つかったか（仕様書7.3対策3）


async def generate_answer(
    query: str,
    document_type: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> AnswerResult:
    """検索結果と会話履歴をコンテキストとしてOllamaに回答をストリーミング生成させる"""
    history: list[dict] = []
    if conversation_id is not None and user_id is not None:
        history = repository.get_recent_messages(conversation_id, user_id, HISTORY_TURNS * 2)  # 5往復＝10件
    # historyはtrim_historyで詰め替わるため、続きの質問かどうかはここで確定させる
    is_continuation = bool(history)

    # 初回質問（履歴なし）は書き換える対象がないためスキップし、LLM呼び出しを節約する（仕様書7.4）
    search_query, is_query_rewritten = await rewrite_query(query, history) if is_continuation else (query, False)

    nodes = search(search_query, document_type=document_type)
    # 関連度が閾値に満たない場合は参照文書を渡さない。根拠のない回答を防ぐ（仕様書7.3対策3）
    is_relevant = has_relevant_document(nodes)
    if not is_relevant:
        nodes = []

    context = "\n\n".join(
        f"[{node.metadata.get('chapter_title') or node.metadata.get('file_path')}]\n{node.text}"
        for node in (n.node for n in nodes)
    )
    # 書き換えは検索用。プロンプトには履歴も同梱されるため、質問は原文のまま渡す
    system_prompt = SYSTEM_PROMPT if is_relevant else NO_DOCUMENT_PROMPT
    user_content = f"参照文書:\n{context}\n\n質問: {query}" if is_relevant else f"質問: {query}"

    # 予算を超える分は古い往復から落とす（仕様書7.4）
    history, is_history_trimmed = trim_history(history, system_prompt + user_content)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        *(ChatMessage(role=MessageRole(m["role"]), content=m["content"]) for m in history),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]
    return AnswerResult(
        response_stream=await get_llm().astream_chat(messages),
        nodes=nodes,
        is_history_trimmed=is_history_trimmed,
        is_query_rewritten=is_query_rewritten,
        is_first_question=not is_continuation,
        has_relevant_document=is_relevant,
    )
