# RAG PoC

ドキュメント（設計書・議事録・マニュアル・仕様書）を対象にしたRAGシステム。
詳細仕様は [`doc/仕様書.md`](doc/仕様書.md) を参照。
進め方のルールは [`CLAUDE.md`](CLAUDE.md)、実装状況は [`実装手順.md`](実装手順.md) を参照。
RAG・LLMの基礎知識は [`doc/RAG_LLM基礎知識.md`](doc/RAG_LLM基礎知識.md) を参照。

## 手順書

| 目的 | 手順書 |
|---|---|
| システム全体の繋がりを図で把握する | [`doc/構成図.md`](doc/構成図.md) |
| AWS（EC2 GPU）へデプロイする | [`doc/デプロイ手順_AWS_EC2_GPU.md`](doc/デプロイ手順_AWS_EC2_GPU.md) |
| オンプレミス環境へデプロイする | [`doc/デプロイ手順_オンプレミス.md`](doc/デプロイ手順_オンプレミス.md) |
| 取り込み済みドキュメントを入れ替える | [`doc/ドキュメント再取り込み手順.md`](doc/ドキュメント再取り込み手順.md) |

### 起動方法の使い分け

| 環境 | 設定ファイル | 起動コマンド |
|---|---|---|
| 開発（Devコンテナ） | `docker-compose.yml` + `docker-compose.override.yml` | VS Codeの「実行とデバッグ」から起動 |
| 本番（CPU） | `docker-compose.prod.yml` | `docker compose -f docker-compose.prod.yml up -d` |
| 本番（GPU） | `docker-compose.prod.yml` + `docker-compose.gpu.yml` | `docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml up -d` |

## ドキュメントの更新方法（Ingestion）

対応拡張子（仕様書4章）：

| 拡張子 | 想定文書 | 対応状況 |
|---|---|---|
| `.md` | 仕様書 | 対応済み |
| `.csv` | 議事録・Q&A集等 | 未対応（TODO） |
| `.docx` | 設計書・マニュアル | 未対応（TODO） |
| `.xlsx` | 一覧表・パラメータ表 | 未対応（TODO） |

1. **ディレクトリにファイルを配置する**
   `document_type`は配置先ディレクトリの1階層目の名前から自動判定される（仕様書4章・5章）。
   ```
   docs/
   ├── specs/       # 仕様書（.md） → document_type="specs"
   ├── design/      # 設計書（.docx想定、現状未対応） → document_type="design"
   ├── minutes/     # 議事録（.csv想定、現状未対応） → document_type="minutes"
   └── manuals/     # マニュアル（.docx想定、現状未対応） → document_type="manuals"
   ```

2. **`run()`を実行する**
   現時点ではCLIエントリポイントは無く、Pythonから直接呼び出す。
   ```bash
   python3 -c "
   from pathlib import Path
   from src.infrastructure.ingestion.ingest_documents import run
   run(Path('docs'))
   "
   ```

3. **差分のみが処理される**（仕様書6.2・6.3）
   - ファイルのmtimeと`.ingestion_state.json`の前回実行時刻を比較し、変更・追加されたファイルのみ処理する
   - 既存ファイルを更新した場合、そのファイルの既存チャンクをQdrantから削除してから、新しい内容で登録し直す（重複しない）
   - 全ファイルを強制的に再取り込みしたい場合は、下記「データのリセット」を先に行う

## データのリセット

Qdrantに投入したデータ（モックデータ等）を消してまっさらな状態に戻す場合、以下の2つを**セットで**リセットする。
片方だけ消すと、差分判定（`.ingestion_state.json`の前回実行時刻）により、Qdrantが空でもファイルが再投入されない場合がある。

1. **Qdrantのコレクション削除**
   ```python
   from src.infrastructure.qdrant_vector_store import get_vector_store, COLLECTION_NAME
   vs = get_vector_store()
   vs.client.delete_collection(COLLECTION_NAME)
   ```
   コレクションは次に`vector_store.add()`が呼ばれた時点で自動的に再作成される。

2. **Ingestion状態ファイルの削除**
   ```bash
   rm -f .ingestion_state.json
   ```

この2つをリセットした後、`ingest_documents.py`の`run(docs_dir)`を実行すれば、`docs_dir`配下の全Markdownファイルが初回実行扱いで再投入される。

## LLMモデルサイズとメモリ要件

現在は`qwen3:8b`（Q4量子化）を使用しており、実行には**約10.6GBのメモリ**が必要（開発環境でDocker Desktopのメモリ割り当てが6.2GBの状態では起動エラーになることを確認済み）。

環境のメモリ制約が厳しい場合、同じQwen3ファミリー内でより小さいモデルに変更できる。

| モデル | パラメータ数 | 備考 |
|---|---|---|
| `qwen3:8b`（現状） | 約82億 | 実測10.6GB必要 |
| `qwen3:4b` | 約40億 | 8Bのおよそ半分程度のメモリと推定（未検証） |
| `qwen3:1.7b` | 約17億 | さらに軽量（未検証） |
| `qwen3:0.6b` | 約6億 | 非常に軽量だが精度はかなり落ちる（未検証） |

**注意**：`qwen3:8b`は仕様書3章で「本番サーバがGPU無し・非ハイスペック前提のため、CPU推論でも現実的なサイズとして8Bに決定」と、精度とCPU実行可能性のバランスを検討した上で選定されたモデル。より小さいモデルに変更すると、日本語での回答品質やハルシネーション対策（システムプロンプトへの忠実さ）が低下する可能性があるため、変更する場合は精度への影響を確認すること。

変更する場合は、Ollamaへの`pull`（`/api/pull`にモデル名を指定）と、[`src/infrastructure/ollama_client.py`](src/infrastructure/ollama_client.py)の`MODEL_NAME`の変更が必要。

**変更の難易度**：実装自体は簡単（上記2点のみ、Embedding・Qdrant側の変更は不要）。FastAPIは`--reload`で自動反映される。
難しいのは実装ではなく、**変更後も回答品質が十分か**（特に日本語での指示追従性・ハルシネーション対策）を確認する部分。変更する際は、同じ質問セットで変更前後の回答を比較すること。
