import json
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DOCUMENT_TYPES = {"未指定": None, "仕様書": "specs", "設計書": "design", "議事録": "minutes", "マニュアル": "manuals"}


def _headers(user_name: str) -> dict:
    """利用者識別ヘッダ。認証追加後もUI側はこの1か所だけの変更で済む（仕様書7.4）"""
    return {"X-User-Id": user_name}


def fetch_conversations(user_name: str) -> list[dict]:
    """会話一覧を更新日時の降順で取得する"""
    return requests.get(f"{API_BASE_URL}/conversations", headers=_headers(user_name), timeout=10).json()


def create_conversation(user_name: str) -> str:
    """会話を新規作成し、会話IDを返す"""
    response = requests.post(f"{API_BASE_URL}/conversations", headers=_headers(user_name), timeout=10)
    return response.json()["id"]


def fetch_messages(user_name: str, conversation_id: str) -> list[dict]:
    """過去の会話のメッセージを取得する。sourcesも含まれる"""
    return requests.get(
        f"{API_BASE_URL}/conversations/{conversation_id}", headers=_headers(user_name), timeout=10
    ).json()


def delete_conversation(user_name: str, conversation_id: str) -> None:
    """会話を削除する"""
    requests.delete(f"{API_BASE_URL}/conversations/{conversation_id}", headers=_headers(user_name), timeout=10)


def render_sources(sources: list[dict]) -> None:
    """参照ソースを折りたたみで表示する。過去の会話を開き直した際も同じ見た目で再現する"""
    if not sources:
        return
    with st.expander(f"参照ソース（{len(sources)}件）"):
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**{source['chapter_title'] or '（章タイトルなし）'}**")
                st.caption(f"{source['file_path']} ・ 関連度 {source['score']:.3f}")

st.set_page_config(page_title="RAG-PoC")

# Claude.aiの見た目に寄せる：アバター非表示、ユーザーは右寄せピル、
# アシスタントは吹き出しなしのプレーンテキスト、入力欄は角丸の大きめボックス
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Hiragino Kaku Gothic ProN", "Hiragino Sans", Meiryo, sans-serif;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none;
    }
    .stChatMessage {
        background-color: transparent;
        padding: 0;
    }
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        justify-content: flex-end;
    }
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background-color: #F0EEE6;
        border-radius: 20px;
        padding: 10px 18px;
        width: fit-content;
        max-width: 70%;
        margin: 0 0 0 auto;
        flex: none;
    }
    [data-testid="stStatusWidgetRunningIcon"] {
        display: none;
    }
    .stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background-color: transparent;
        padding: 0;
    }
    [data-testid="stChatInput"] {
        border-radius: 32px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }
    .loading-dots {
        display: flex;
        gap: 6px;
        padding: 8px 0;
    }
    .loading-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #A8A296;
        animation: bounce 1.4s infinite ease-in-out both;
        display: inline-block;
    }
    .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

with st.sidebar:
    st.title("RAG-PoC")
    st.caption("社内ドキュメントに基づいて質問に回答します")

    user_name = st.text_input("利用者名", placeholder="例：tanaka")

    if st.button("＋ 新しい会話", use_container_width=True):
        # 会話の作成は最初の質問時。ここでは表示を空にするだけ（空の会話を一覧に残さない）
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    st.divider()

    if user_name:
        st.subheader("会話履歴")
        for conversation in fetch_conversations(user_name):
            is_current = conversation["id"] == st.session_state.conversation_id
            open_column, delete_column = st.columns([5, 1])
            if open_column.button(
                conversation["title"],
                key=f"open_{conversation['id']}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.conversation_id = conversation["id"]
                st.session_state.messages = fetch_messages(user_name, conversation["id"])
                st.rerun()
            if delete_column.button("🗑", key=f"delete_{conversation['id']}"):
                delete_conversation(user_name, conversation["id"])
                if is_current:
                    st.session_state.messages = []
                    st.session_state.conversation_id = None
                st.rerun()

        st.divider()

    st.subheader("絞り込み")
    document_type_label = st.selectbox("文書種別", list(DOCUMENT_TYPES.keys()))
    document_type = DOCUMENT_TYPES[document_type_label]

if not st.session_state.messages:
    st.markdown(
        "<h2 style='text-align:center; margin-top: 12vh; font-weight:600;'>今日はどうしましたか？</h2>",
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources") or [])

# st.chat_inputはタブ等のコンテナ内に置くと画面下部への自動固定が効かないため、
# トップレベルで呼び出す（Streamlitの仕様）
query = st.chat_input(
    "質問を入力してください" if user_name else "サイドバーで利用者名を入力してください",
    disabled=not user_name,
)
if query:
    # 会話は最初の質問時に作成する（ボタンを押しただけの空の会話を一覧に残さないため）
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = create_conversation(user_name)
    is_continuation = bool(st.session_state.messages)  # 案内の要否判定に使うため追加前に確定させる

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        payload = {
            "query": query,
            "document_type": document_type,
            "conversation_id": st.session_state.conversation_id,
        }
        with requests.post(
            f"{API_BASE_URL}/query", json=payload, headers=_headers(user_name), stream=True, timeout=300
        ) as resp:
            lines = resp.iter_lines()

            # sourcesは検索完了時点で確定済みの1行目。LLM生成が始まるまでの待ち時間を
            # ここでローディング表示してカバーする（トークン到着後はwrite_stream自体が進捗を示す）
            sources = []
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                '<div class="loading-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )
            for line in lines:
                if not line:
                    continue
                data = json.loads(line)
                if data["type"] == "sources":
                    sources = data["sources"]
                    break

            # ソース受信後もLLMの最初のトークンが届くまで待ち時間があるため、
            # ローディング表示は最初のトークン到着時に消す
            done: dict = {}

            def token_generator():
                is_first_token = True
                for line in lines:
                    if not line:
                        continue
                    data = json.loads(line)
                    if data["type"] == "token":
                        if is_first_token:
                            loading_placeholder.empty()
                            is_first_token = False
                        yield data["content"]
                    elif data["type"] == "done":
                        done.update(data)  # 最終行。切り詰め・書き換え・タイトルの通知

            answer = st.write_stream(token_generator())

        render_sources(sources)

        if done.get("is_history_trimmed"):
            st.info("会話が長くなったため、古いやり取りを一部省略して回答しました。")
        # 書き換えなしは「自己完結した質問」でも起きるため、回答が「分かりません」の場合だけ案内する
        if is_continuation and not done.get("is_query_rewritten") and "分かりません" in answer:
            st.warning("前の質問との関連が判断できませんでした。具体的な名称を含めて質問し直してください。")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    if done.get("title"):
        st.rerun()  # 生成されたタイトルを会話一覧に反映する
