# デプロイ手順：AWS EC2（GPU）

購買システム向けRAGシステムを、GPU搭載のEC2インスタンス1台にデプロイする手順。

> **前提**：本番用の設定ファイル（`docker-compose.prod.yml` / `docker-compose.gpu.yml`）は
> リポジトリに用意済み。4章はそれらの中身の説明と、GPU利用時に追加で必要な変更点をまとめている。
>
> 開発用の `docker-compose.yml` は Devコンテナ専用（apiを `sleep infinity` で待機させる構成）で、
> 本番では使わない。

---

## 0. まず全体像

RAGシステムは4つの部品が連携して動く。1台のEC2の中で、それぞれDockerコンテナとして動かす。

```text
利用者のブラウザ
      ↓ 8501番ポート
[ streamlit ] ← 画面
      ↓ 8000番ポート
[ api ]       ← FastAPI。検索と回答生成の司令塔
   ↓ 6333       ↓ 11434
[ qdrant ]    [ ollama ]
 ベクトルDB     LLM（Qwen3 8B）
```

GPUを使うのは **ollama（LLM推論）** と、任意で **api（BGE-M3のEmbedding計算）** である。
qdrantとstreamlitはGPUを使わない。

---

## 1. インスタンスを選ぶ

### 推奨スペック

| 項目 | 推奨 | 最小 | 理由 |
|---|---|---|---|
| インスタンスタイプ | `g5.xlarge`（A10G 24GB） | `g4dn.xlarge`（T4 16GB） | Qwen3 8Bは4bit量子化で約5.2GB。BGE-M3を同居させても16GBに収まる |
| ストレージ（EBS） | 100GB gp3 | 60GB gp3 | モデル約8GB＋Dockerイメージ＋ドキュメント＋OSの余裕 |
| OS | Ubuntu 22.04 LTS | 同左 | NVIDIA関連の情報が最も揃っている |
| AMI | Deep Learning Base OSS Nvidia Driver AMI | Ubuntu標準AMI＋ドライバ手動導入 | Deep Learning AMIならNVIDIAドライバが導入済みで手間が減る |

> **なぜGPUメモリの話が出るのか**：LLMは全パラメータをメモリに載せて計算する。
> Qwen3 8Bは80億パラメータあるが、Ollamaは既定で4bitに圧縮（量子化）して読み込むため約5.2GBで済む。
> GPUメモリがモデルサイズより小さいと、CPUメモリへの退避が起きて極端に遅くなる。

### コストの目安（東京リージョン `ap-northeast-1`・オンデマンド・Linux）

| インスタンス | GPU | 円/時 | 24時間×30日 | 平日9〜19時（月200時間） |
|---|---|---|---|---|
| `g4dn.xlarge` | T4 16GB | 109円 | **78,500円** | **21,800円** |
| `g6.xlarge` | L4 24GB | 179円 | 129,100円 | 35,900円 |
| `g5.xlarge` | A10G 24GB | 224円 | 161,300円 | 44,800円 |
| （参考）`m6i.2xlarge` | GPUなし・8vCPU/32GB | 76円 | 54,900円 | 15,200円 |

上記に加えてEBS（100GB gp3）が月1,500円程度かかる。
**インスタンスを停止すればインスタンス料金は発生せず、EBS代のみになる。**
検証中は使わない時間帯に停止する運用を強く勧める。

> **Qwen3 8Bには `g4dn.xlarge`（T4 16GB）で足りる。** モデルは4bit量子化で約5.2GBのため、
> 16GBのGPUメモリに収まる。BGE-M3（約2.2GB）を同居させても余裕がある。
> 1章の推奨は余裕を見た構成であり、コストを優先するなら `g4dn.xlarge` から始めてよい。

> **金額の根拠**：インスタンス単価は東京リージョンのオンデマンド価格（Linux）、
> 為替は 1米ドル＝153.594円（2026年9月10日時点）で換算した。
> AWSの料金は米ドル建てのため為替で変動する。
> 実際の金額は[AWS公式の料金ページ](https://aws.amazon.com/jp/ec2/pricing/on-demand/)と
> AWS Pricing Calculator で確認すること。
> 1年以上の継続利用が決まっている場合は、Savings Plans や リザーブドインスタンスで
> 3〜4割程度安くなる。

### さくらのクラウドとの比較

国内クラウドを検討する場合の参考。

| | プラン | 円/時 | 月額 |
|---|---|---|---|
| さくらのクラウド | 高火力VRT V100 | 481円 | 231,000円 |
| さくらのクラウド | 高火力VRT H100 | 990円 | 385,000円 |
| さくらのクラウド | 8コア/32GB（GPUなし） | 145円 | 29,040円 |

**GPUを使うならAWSの方が安い。** さくらのGPUはV100とH100の2択で、
どちらもQwen3 8Bには過剰である。AWSの `g4dn.xlarge`（78,500円/月）と比べると、
さくらの最小GPU構成（V100・231,000円/月）は約3倍になる。
なお**V100は2027年3月31日で提供終了**が告知されているため、今から選ぶならH100になる。

**CPU運用ならさくらの方が安い。** 8コア/32GBで29,040円/月と、
AWSの同等構成（`m6i.2xlarge`・54,900円/月）のおよそ半額である。
仕様書10.1が想定する「GPU無しの本番環境」であれば、さくらが有力な選択肢になる。

課金体系の違いにも注意する。さくらは20日未満なら日割りだが、それ以降は月額で頭打ちになる。
AWSは純粋な従量制で、停止すれば課金も止まる。
**24時間フル稼働ならさくらの月額固定が効き、断続的な利用ならAWSが有利**である。

（さくらの料金は[さくらのクラウド公式ページ](https://cloud.sakura.ad.jp/products/server/gpu/)より）

---

## 2. インスタンスを起動してアクセスする

### 2.1 セキュリティグループ

**このシステムには認証機能が無い**（仕様書13章でスコープ外と決定）。
URLを知っている人は誰でも全ドキュメントを検索できてしまうため、**ネットワークで守る必要がある**。

| ポート | 用途 | 公開範囲 |
|---|---|---|
| 22 | SSH | 管理者のIPアドレスのみ |
| 8501 | Streamlit（利用者の画面） | **社内のグローバルIPのみ**。`0.0.0.0/0` は絶対に避ける |
| 8000 | FastAPI | 公開しない（同一インスタンス内からのみ） |
| 6333 | Qdrant | 公開しない |
| 11434 | Ollama | 公開しない |

8000・6333・11434 はコンテナ同士の通信にしか使わないため、セキュリティグループで開ける必要はない。

### 2.2 SSHで接続

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2のパブリックIP>
```

---

## 3. サーバ側の下準備

### 3.1 GPUドライバの確認

```bash
nvidia-smi
```

GPU名とメモリ量が表示されれば導入済み。`command not found` の場合はドライバが入っていないので、
Deep Learning AMIで作り直すか、NVIDIAドライバを導入する。

### 3.2 Docker と Docker Compose

```bash
# Dockerの導入
curl -fsSL https://get.docker.com | sudo sh

# sudoなしでdockerを使えるようにする（実行後、一度ログアウトして入り直す）
sudo usermod -aG docker $USER
```

### 3.3 NVIDIA Container Toolkit（重要）

**これが無いとコンテナからGPUが見えない。** ホストにドライバがあるだけでは不十分である。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**動作確認**（ここでGPU情報が出れば準備完了）：

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 3.4 ソースコードを配置

```bash
git clone <リポジトリのURL> ~/rag-support-system
cd ~/rag-support-system
```

---

## 4. 本番用ファイルの構成と、GPU利用時に必要な変更

### 4.1 【用意済み】`docker-compose.prod.yml` と `docker-compose.gpu.yml`

本番用のCompose設定は2本立てになっている。**CPU環境でもGPU環境でも同じ基本ファイルを使い、
GPU環境では上書きファイルを重ねる**という、Composeの標準的な構成である。

| ファイル | 役割 |
|---|---|
| `docker-compose.prod.yml` | 本番の基本構成（CPU前提）。api・streamlit・qdrant・ollama の4サービス |
| `docker-compose.gpu.yml` | GPU用の上書き。api と ollama にGPU割り当てを追加し、`EMBEDDING_DEVICE=cuda` にする |

GPU環境では、両方を `-f` で並べて指定する（後に書いた方が優先される）。

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml up -d
```

**開発用の `docker-compose.yml` との違い**

| 項目 | 開発用（`docker-compose.yml`） | 本番用（`docker-compose.prod.yml`） |
|---|---|---|
| api の起動 | `sleep infinity`（VS Codeから中に入って手動起動） | `uvicorn` で自動起動 |
| streamlit | サービス定義なし（launch.jsonから起動） | サービスとして定義 |
| ソースコード | `.:/workspace` でホストをマウント | イメージに焼き込む |
| 公開ポート | 8000/8501 をホストへ公開 | 8501 のみ |
| 再起動 | なし | `restart: unless-stopped` |

`docs/` だけは本番でもホスト側をマウントしている（読み取り専用）。
ドキュメントを差し替えるたびにイメージを作り直さずに済むようにするためである。

### 4.2 【用意済み】`Dockerfile`

本番用にソースコードをイメージへ焼き込む形にしてある。
開発（Devコンテナ）ではホストの `.` が `/workspace` にマウントされて上書きされるため、
同じ Dockerfile を開発でもそのまま使える。

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

# コンテナ自体が隔離環境のため.venvは作らず、システムPythonに直接インストールする
ENV UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /workspace

# 依存関係を先にインストールする（コード変更のたびに再インストールしないため）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# アプリケーションのコードを焼き込む
COPY src/ ./src/
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/
```

### 4.3 【要変更】`pyproject.toml` ― GPU利用時の最重要ポイント

**このファイルだけは、デプロイ時に手で変更する必要がある。**

リポジトリの既定はCPU専用版のPyTorchになっている。開発環境（CPUのDevコンテナ）を壊さないため、
CUDA版はコメントアウトした状態で併記してある。

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]     # ← CPU専用のPyTorchを取得している

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

PyTorchには「CPU専用版」と「CUDA対応版」があり、既定は明示的にCPU専用版を指定している。
CPU専用版は `torch.cuda.is_available()` が常に `False` を返すため、GPUがあっても使われない。

GPUで動かす場合は、**上のCPU指定をコメントアウトし、ファイル内に用意されている
CUDA版の指定（コメントアウト済み）を有効にする**：

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

差し替えたら `uv lock` を実行して `uv.lock` を更新すること（Dockerfileが `--frozen` で
ロックファイルと突き合わせるため、更新しないとビルドが失敗する）。

> **注意**：CUDA版のPyTorchはCPU版よりかなり大きい（2GB超）。イメージのビルド時間とEBS消費が増える。
> また、EC2のドライバが対応するCUDAバージョンに合わせる必要がある（`nvidia-smi` の右上に表示される
> CUDA Version が 12.4 以上なら `cu124` で問題ない）。
>
> **Ollama（LLM推論）はこの設定と無関係にGPUを使う。** Ollamaは自前のランタイムを持つため、
> PyTorchをCPU版のままにしても、LLMの回答生成はGPUで高速化される。
> つまり「まずはLLMだけGPU化し、Embeddingは後で」という段階的な進め方も可能である。

### 4.4 【用意済み】`src/infrastructure/llama_index_factory.py`

Embeddingを動かすデバイスは、環境変数 `EMBEDDING_DEVICE` で切り替わるようにしてある
（既定は `cpu`）。`docker-compose.gpu.yml` がこれを `cuda` に設定するため、
GPU環境ではコードを変更する必要はない。

```python
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
USE_FP16 = EMBEDDING_DEVICE.startswith("cuda")  # fp16はGPUでのみ有効
```

`use_fp16` はCPUでは無効（むしろ遅くなる）でGPUでのみ有効なため、デバイスに連動させている。

**ただし、4.3のPyTorch差し替えを行っていない場合、この設定を `cuda` にしても動作しない。**
CPU専用版のPyTorchはCUDAデバイスを認識できないためである。

### 4.5 【確認】`docker-compose.override.yml` を持ち込まないこと

このファイルは **macOSの開発環境専用**（ホストのOllamaに接続してMetalを使うための設定）である。
本番サーバにこのファイルがあると、`docker compose` が自動で読み込んで
`OLLAMA_BASE_URL=http://host.docker.internal:11434` を上書きしてしまい、**Ollamaに接続できなくなる**。

本番では常に `-f` でファイルを明示して起動する（5章のコマンド参照）。
`-f` を明示した場合、`docker-compose.override.yml` は自動読み込みされないため安全である。

### 4.6 【用意済み】`.ingestion_state.json` の置き場所

前回取り込み時刻を記録するファイルの場所は、環境変数 `INGESTION_STATE_FILE` で
変更できるようにしてある（既定はプロジェクトルート直下の `.ingestion_state.json`）。

本番ではコードをイメージに焼き込むため、既定のままだとコンテナを作り直すたびに状態が消え、
毎回すべてのドキュメントが再取り込みされてしまう（動作はするが時間がかかる）。
`docker-compose.prod.yml` では `ingestion_state` ボリュームを `/workspace/state` にマウントし、
`INGESTION_STATE_FILE=/workspace/state/.ingestion_state.json` を設定済みである。

---

## 5. デプロイの実行

### 5.1 イメージのビルドと起動

```bash
cd ~/rag-support-system

# ビルド（CUDA版PyTorchを含むため10〜20分かかることがある）
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml build

# 起動
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml up -d

# 状態確認（すべて Up になっていること）
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml ps
```

### 5.2 LLMモデルの取得

**初回のみ必要。** Ollamaのコンテナは起動しただけではモデルを持っていない。

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml exec ollama ollama pull qwen3:8b
```

5.2GBのダウンロードのため数分かかる。`ollama_data` ボリュームに保存されるため、
コンテナを作り直しても再ダウンロードは不要である。

### 5.3 GPUが使われているかの確認

```bash
# Ollamaに簡単な質問を投げながら、別のターミナルでGPU使用率を見る
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml exec ollama ollama run qwen3:8b "こんにちは"

# 別ターミナルで
nvidia-smi
```

`nvidia-smi` の下部のプロセス一覧に `ollama` が現れ、GPUメモリが数GB使われていればGPU推論が効いている。

### 5.4 ドキュメントの取り込み

```bash
# ホスト側の docs/ 配下に正規のドキュメントを配置してから
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml exec api \
  python -c "from pathlib import Path; from src.infrastructure.ingestion.ingest_documents import run; run(Path('docs'))"
```

取り込み手順の詳細と、入れ直しの方法は
[`ドキュメント再取り込み手順.md`](ドキュメント再取り込み手順.md) を参照。

### 5.5 動作確認

```bash
# Qdrantに登録されたチャンク数を確認（0でなければ取り込み成功）
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml exec api \
  python -c "import urllib.request, json; print(json.load(urllib.request.urlopen('http://qdrant:6333/collections/documents'))['result']['points_count'])"
```

ブラウザで `http://<EC2のパブリックIP>:8501` を開き、
サイドバーに利用者名を入力してから質問すれば画面から確認できる。

---

## 6. 運用

### ログを見る

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml logs -f api        # APIのログ
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml logs -f streamlit  # 画面側のログ
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml logs -f ollama     # LLMのログ
```

### 再起動・停止

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml restart api   # APIだけ再起動
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml down          # 全停止（データは残る）
```

> **`down -v` は絶対に使わないこと。** `-v` を付けるとボリュームごと削除され、
> 取り込み済みのベクトルデータ・会話履歴・ダウンロード済みモデルがすべて消える。

### コードを更新したとき

```bash
git pull
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml build api streamlit
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml up -d api streamlit
```

### バックアップ対象

| 対象 | 中身 | 消えたときの影響 |
|---|---|---|
| `qdrant_storage` ボリューム | ベクトル化されたドキュメント | 再取り込みで復旧可能（時間はかかる） |
| `app_data` ボリューム | 会話履歴のSQLite | **復旧不可**。過去の会話が失われる |
| `ollama_data` ボリューム | LLMモデル | 再ダウンロードで復旧可能 |
| `hf_cache` ボリューム | Embeddingモデル | 再ダウンロードで復旧可能 |

会話履歴だけは復旧手段が無いため、定期的なバックアップを推奨する。

---

## 7. つまずきやすい点

| 症状 | 原因と対処 |
|---|---|
| `docker: Error response ... could not select device driver "nvidia"` | NVIDIA Container Toolkit が未導入。3.3を実施する |
| 回答生成が異常に遅い（1分以上） | GPUが使われていない。`nvidia-smi` でollamaプロセスを確認。5.3を参照 |
| `Connection refused` でOllamaにつながらない | `docker-compose.override.yml` を本番に持ち込んでいないか確認（4.5） |
| 質問しても必ず「関連する文書が見つかりません」 | ドキュメント未取り込み。5.4を実施。5.5でチャンク数が0でないか確認 |
| ブラウザから8501につながらない | セキュリティグループで8501が許可されているか、`--server.address 0.0.0.0` が付いているか確認 |
| メモリ不足でコンテナが落ちる | BGE-M3がdense用・sparse用の2つ常駐し約4.5GB使う。インスタンスのRAMを確認する |
