"""関連度スコアの閾値を実測するスクリプト（仕様書7.3対策3・9章）。

検索のみを行いLLMは呼ばないため、Qdrantが起動していれば数十秒で完了する。
ドキュメントを入れ替えたら（コーパスが変われば特にsparseのスコア分布が動くため）
そのたびに実行し、仕様書7.3の閾値と測定根拠を更新すること。

実行方法：
    python scripts/measure_thresholds.py
"""

from src.application.rag_service import (
    DENSE_RELEVANCE_THRESHOLD,
    SPARSE_RELEVANCE_THRESHOLD,
    has_relevant_document,
    search,
)
from src.infrastructure.qdrant_vector_store import DENSE_SCORE_KEY, SPARSE_SCORE_KEY

# 質問は6文書すべてに散るよう配分する。特定の文書に偏ると、その偏りが閾値に紛れ込むため
RELEVANT_QUESTIONS = [
    "発注の消費税はどのように計算されますか",
    "発注番号はどのような規則で採番されますか",
    "1000万円以上の発注には誰の承認が必要ですか",
    "代理承認者は最長何日間設定できますか",
    "取引先の与信限度額の初期値はいくらですか",
    "会計システムへの連携は何時に実行されますか",
    "一時保存した発注が見つからない場合はどうすればよいですか",
]
IRRELEVANT_QUESTIONS = [
    "年末調整の手続きを教えてください",
    "有給休暇はどうやって申請しますか",
    "会議室の予約方法を教えてください",
    "経費精算の締め日はいつですか",
    "社員食堂の今日のメニューは何ですか",
]
GREETING_QUESTIONS = [
    "こんにちは",
    "ありがとうございます",
    "お疲れ様です",
]
# (群の名前, 質問リスト, 関連ありと判定されるべきか)
QUESTION_GROUPS = [
    ("関連あり", RELEVANT_QUESTIONS, True),
    ("無関係", IRRELEVANT_QUESTIONS, False),
    ("挨拶", GREETING_QUESTIONS, False),
]


def measure(query: str) -> tuple[float, float, bool]:
    """1問を検索し、(dense最大値, sparse最大値, 関連ありと判定されたか)を返す。

    has_relevant_documentは結果ノードのいずれかが閾値を超えれば関連ありとみなすため、
    1位のスコアではなく結果全体の最大値が判定を左右する
    """
    nodes = search(query)
    dense_scores = [n.node.metadata.get(DENSE_SCORE_KEY) or 0.0 for n in nodes]
    sparse_scores = [n.node.metadata.get(SPARSE_SCORE_KEY) or 0.0 for n in nodes]
    return max(dense_scores, default=0.0), max(sparse_scores, default=0.0), has_relevant_document(nodes)


def main() -> None:
    print(f"現在の閾値: dense >= {DENSE_RELEVANCE_THRESHOLD} または sparse >= {SPARSE_RELEVANCE_THRESHOLD}\n")

    mistakes: list[str] = []
    for group_name, questions, should_be_relevant in QUESTION_GROUPS:
        print(f"【{group_name}】（{len(questions)}問・関連あり判定が期待値={should_be_relevant}）")
        denses: list[float] = []
        sparses: list[float] = []
        for query in questions:
            dense, sparse, is_relevant = measure(query)
            denses.append(dense)
            sparses.append(sparse)
            mark = "OK " if is_relevant == should_be_relevant else "NG "
            if is_relevant != should_be_relevant:
                mistakes.append(f"{group_name}: {query}")
            print(f"  {mark} dense={dense:.4f} sparse={sparse:.4f}  {query}")
        print(f"  → dense {min(denses):.4f}〜{max(denses):.4f} / sparse {min(sparses):.4f}〜{max(sparses):.4f}\n")

    total = sum(len(q) for _, q, _ in QUESTION_GROUPS)
    print(f"判定結果: {total}問中 {total - len(mistakes)}問が期待通り")
    for mistake in mistakes:
        print(f"  NG: {mistake}")


if __name__ == "__main__":
    main()
