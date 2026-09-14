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

### この手順書の流れ

| 章 | やること |
|---|---|
| 1 | インスタンスのスペックとコストを決める |
| 2 | EC2インスタンスを起動し、SSHで入れる状態にする |
| 3 | サーバ側の下準備（ドライバ・Docker・Toolkit・ソース配置） |
| 4 | 本番用ファイルの確認 |
| 5 | ビルド・起動・モデル取得・ドキュメント取り込み |
| 6 | 運用（ログ・再起動・バックアップ） |
| 7 | つまずきやすい点 |

**4章に手作業が必要なファイルは無い。** PyTorchのCUDA版への切り替えも自動化されている（4.3）。
用意済みのファイルをそのまま使える。

---

## 1. インスタンスを選ぶ

### 推奨スペック

| 項目 | 値 | 理由 |
|---|---|---|
| インスタンスタイプ | `g4dn.xlarge`（T4 16GB） | Qwen3 8Bは4bit量子化で約5.2GB。BGE-M3（約2.2GB）を同居させても16GBに収まる |
| ストレージ（EBS） | 100GB gp3 | モデル約8GB＋Dockerイメージ＋ドキュメント＋OSの余裕 |
| OS | Ubuntu 22.04 LTS | NVIDIA関連の情報が最も揃っている |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) | ドライバ・CUDA・Docker・NVIDIA Container Toolkitが導入済みで、3章の下準備をほぼ省略できる（選び方は2.3） |

> **なぜGPUメモリの話が出るのか**：LLMは全パラメータをメモリに載せて計算する。
> Qwen3 8Bは80億パラメータあるが、Ollamaは既定で4bitに圧縮（量子化）して読み込むため約5.2GBで済む。
> GPUメモリがモデルサイズより小さいと、CPUメモリへの退避が起きて極端に遅くなる。

### コストの目安（東京リージョン `ap-northeast-1`・オンデマンド・Linux）

| インスタンス | GPU | 円/時 | 24時間×30日 | 平日9〜19時（月200時間） |
|---|---|---|---|---|
| `g4dn.xlarge` | T4 16GB | 109円 | **78,500円** | **21,800円** |

上記に加えてEBS（100GB gp3）が月1,500円程度かかる。
**インスタンスを停止すればインスタンス料金は発生せず、EBS代のみになる。**
検証中は使わない時間帯に停止する運用を強く勧める。

---

## 2. インスタンスを起動してアクセスする

### 2.0 この章の全体像

1章で決めたスペックを、実際にEC2のコンソールで形にする章である。
**起動ボタンを押した瞬間から課金が始まる**ため、押す前に決めておくことと、
押した後でしか確認できないことを分けて並べている。

```text
【事前】 2.1  GPUインスタンスのvCPUクォータを確認
              ↓ 上限が 0 のままなら引き上げ申請（承認に数時間〜数日）
【起動】 2.2  リージョンを東京に合わせる
         2.3  AMIを選ぶ           … Deep Learning Base OSS Nvidia Driver GPU AMI
         2.4  インスタンスタイプ  … g4dn.xlarge
         2.5  キーペアを作る      … .pem をダウンロード（再取得不可）
         2.6  ネットワーク設定    … SSH(22) を自分のIPだけに開ける
         2.7  ストレージ設定      … 100GB / gp3
         2.8  内容を確認して起動  … ここから課金開始
              ↓
【確認】 2.9  SSHで接続し nvidia-smi が通ることを確認 → 3章へ
【運用】 2.10 停止・再開時の注意（パブリックIPが変わる）
```

**所要時間の目安**：クォータが足りていれば2.2から2.9まで15分程度。
不足していると2.1で数時間〜数日待つことになるため、**2.1を最初に確認する**。

**先に決めておく値**

| 項目 | 本手順での値 |
|---|---|
| リージョン | アジアパシフィック（東京）`ap-northeast-1` |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) |
| インスタンスタイプ | `g4dn.xlarge` |
| ルートボリューム | 100GB / gp3 |
| ログインユーザー名 | `ubuntu` |

---

### 2.1 【事前確認】GPUインスタンスのvCPUクォータ

**新規のAWSアカウントでは、GPUインスタンスの上限が `0` に設定されていることが多い。**
この状態で起動を押しても `VcpuLimitExceeded` というエラーで失敗する。
AMIやスペックの選択とは無関係に弾かれるため、**他のどの作業よりも先に確認する**。

Service Quotas コンソール →「AWS のサービス」→ Amazon Elastic Compute Cloud (Amazon EC2) →
**Running On-Demand G and VR instances**（クォータコード `L-DB2E81BA`）を開く。

CLIで確認する場合：

```bash
aws service-quotas get-service-quota \
  --region ap-northeast-1 \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --query "Quota.Value" --output text
```

**この数値はインスタンスの台数ではなくvCPU数である。**
`g4dn.xlarge` は4vCPUなので **4以上** が必要になる。
不足していれば同じ画面から引き上げを申請する。承認は即時のこともあるが、数日かかる場合もある。

> 申請フォームには用途を記入する欄がある。
> 「社内文書検索システムの検証のため、東京リージョンで g4dn.xlarge を1台起動したい」
> 程度の具体性があれば足りる。

---

### 2.2 リージョンを確認する

マネジメントコンソール右上のリージョン表示が
**アジアパシフィック (東京) ap-northeast-1** になっていることを確認してから、
EC2ダッシュボードの「**インスタンスを起動**」を押す。

リージョンを間違えると、1章のコスト試算（東京リージョン基準）が合わなくなるうえ、
作成したキーペアやセキュリティグループは別リージョンからは見えないため、探し回ることになる。

---

### 2.3 AMI（OSイメージ）を選ぶ

「アプリケーションおよび OS イメージ」の検索欄に **`Deep Learning Base OSS`** と入力し、
以下を選ぶ。

```text
Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
```

**このAMIを選ぶ理由**：NVIDIAドライバ・CUDA・Docker・NVIDIA Container Toolkit が
導入済みで、**3章の下準備（3.1〜3.3）をほぼスキップできる**。
このシステムは4つの部品すべてがDockerコンテナ内で動くため、
ホスト側に必要なものはこの4つに尽きる。

名前のよく似たAMIが並ぶので注意する。

| 名称 | 中身 | 判定 |
|---|---|---|
| Deep Learning **Base OSS Nvidia Driver** GPU AMI | ドライバ＋CUDA＋Docker＋Container Toolkit | **これを選ぶ** |
| Deep Learning **Base Proprietary Nvidia Driver** GPU AMI | 同上（ドライバがプロプライエタリ版） | 不要 |
| Deep Learning **OSS Nvidia Driver AMI GPU PyTorch 2.x** | 上記＋PyTorch・conda一式 | **不要**。PyTorchは4.3でコンテナ内に導入されるため、ホスト側の同梱分は一度も使われない |
| Ubuntu Server 22.04 LTS（通常版） | 何も無し | ドライバ導入から全て手作業になる |

確実にAMI IDを取得したい場合：

```bash
aws ssm get-parameter --region ap-northeast-1 \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query "Parameter.Value" --output text
```

> **同梱されるCUDAのバージョン**：20250205リリース以降、CUDA Toolkit 12.6 が同梱されている
> （CUDA 12.1・12.2は削除済み）。4.3で使うCUDA版PyTorch（`cu126`）とバージョンが一致する。
> 通常のUbuntu AMIから手動でドライバを入れた場合、標準リポジトリでは
> CUDA 12.2相当になることがあり、`cu126` の要件を満たさない。

---

### 2.4 インスタンスタイプを選ぶ

検索欄に `g4dn.xlarge` と入力して選択する。

| 項目 | 値 |
|---|---|
| GPU | NVIDIA T4 / GPUメモリ 16GB |
| vCPU | 4 |
| メモリ（RAM） | 16GiB |

Qwen3 8B（4bit量子化で約5.2GB）とBGE-M3（約2.2GB）を同居させても、
16GBのGPUメモリに収まる。

**RAMが16GiBしかない点には注意する。** BGE-M3はdense用・sparse用の2つが常駐し、
CPU上では合計約4.5GBを消費する（仕様書8章）。
`docker-compose.gpu.yml` により `EMBEDDING_DEVICE=cuda` になればGPUメモリ側に載るが、
CPU運用のまま検証するとRAMが逼迫する。7章の最終行はこの症状である。

> **`InsufficientInstanceCapacity` で起動に失敗する場合**：そのアベイラビリティゾーンに
> GPUインスタンスの空きが無い。ネットワーク設定でサブネットを別のAZに変更して再試行する。

---

### 2.5 キーペアを作成する

EC2にSSHで入るための鍵を作る。**2.8で起動ボタンを押す前に作っておく必要がある**
（起動後にキーペアを後付けすることはできない）。

#### 作成前の準備（Mac側）

鍵の置き場所を先に用意する。`~/.ssh` が無い、または権限が緩いと、
鍵ファイル側の権限を正しく絞っても接続が拒否される。

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

#### コンソールでの操作

インスタンス起動ウィザードの「**キーペア (ログイン)**」欄で「**新しいキーペアの作成**」を押し、
以下を入力する。

| 項目 | 値 |
|---|---|
| キーペア名 | `rag-poc` |
| キーペアのタイプ | **ED25519** |
| プライベートキー形式 | **`.pem`** |

- キーペア名はAWS上での識別名で、リージョン内で一意であればよい。
  ここでは `rag-poc` とし、以降の手順はすべてこの名前を前提にする
- `.ppk` はWindowsのPuTTY用である。macOS・Linux・Windows標準のOpenSSHから接続するなら `.pem` を選ぶ

「キーペアを作成」を押すと、ブラウザが `rag-poc.pem` をダウンロードする。

#### ダウンロードした鍵を配置する

**このファイルは作成時にしかダウンロードできず、後から再取得できない。**
紛失するとそのインスタンスにSSHで入れなくなる（復旧には別インスタンスへのボリューム付け替えが必要になる）。
画面を閉じる前に配置まで済ませる。

保存先はブラウザの設定によって変わるため、まず実際に落ちた場所を確認する。

```bash
ls -l ~/Downloads/rag-poc.pem
```

見つからない場合はブラウザのダウンロード履歴から保存先を確認する。場所が分かったら移動して権限を絞る。

```bash
mv ~/Downloads/rag-poc.pem ~/.ssh/
chmod 400 ~/.ssh/rag-poc.pem
```

権限が緩いままだと、接続時にOpenSSHが
`WARNING: UNPROTECTED PRIVATE KEY FILE!` と警告して接続を拒否する。

---

### 2.6 ネットワーク設定（セキュリティグループ）

**このシステムには認証機能が無い**（仕様書13章でスコープ外と決定）。
URLを知っている人は誰でも全ドキュメントを検索できてしまうため、**ネットワークで守る必要がある**。

最終的に必要な公開範囲は以下のとおり。

| ポート | 用途 | 公開範囲 |
|---|---|---|
| 22 | SSH | 管理者のIPアドレスのみ |
| 8501 | Streamlit（利用者の画面） | **社内のグローバルIPのみ**。`0.0.0.0/0` は絶対に避ける |
| 8000 | FastAPI | 公開しない（同一インスタンス内からのみ） |
| 6333 | Qdrant | 公開しない |
| 11434 | Ollama | 公開しない |

8000・6333・11434 はコンテナ同士の通信にしか使わないため、セキュリティグループで開ける必要はない。

**起動の時点では22番だけを開ける。**
新規セキュリティグループを作成し、SSHのソースに「マイIP」を選ぶ
（作業場所のグローバルIPが自動で入力される）。
8501番は5章でデプロイが完了し、実際に画面を公開する段になってから追加する。
中身が空のうちからポートを開けておく理由はない。

あわせて以下を確認する。

- **パブリックIPの自動割り当て**：「有効化」になっていること（無効だとSSHで到達できない）
- **サブネット**：デフォルトVPCのパブリックサブネットであること

> 「マイIP」は設定した瞬間のグローバルIPを固定値として記録するだけである。
> 社内回線のIPが変わったり、別の場所から作業したりすると接続できなくなる。
> その場合はセキュリティグループのインバウンドルールを編集して現在のIPに更新する。

---

### 2.7 ストレージを設定する

「ストレージを設定」で、
ルートボリュームを **100GB / gp3** にする。

**既定値はAMIによって異なる。** Ubuntu標準AMIは8GiB、Deep Learning系AMIは100GB前後で
提示されることが多い。**実際に表示されている値を確認してから** 100GBに合わせること。

| 項目 | 値 |
|---|---|
| サイズ | 100GB |
| ボリュームタイプ | **gp3**（AMIによってはgp2が選ばれるため明示的に変更する） |

100GBの内訳は、OSとドライバ、Dockerイメージ（CUDA版PyTorchを含むため数GB）、
LLMモデル5.2GB、BGE-M3約2.2GB、ドキュメント、Qdrantのインデックスである。

> **`g4dn.xlarge` に付属する125GBのNVMe（インスタンスストア）は使わないこと。**
> インスタンスを停止すると中身が消える一時領域である。
> Dockerのボリューム（`qdrant_storage` / `app_data` / `ollama_data`）をここに置くと、
> 停止・起動のたびに取り込み済みのベクトルデータと会話履歴が消える。
> 既定のまま何もしなければEBS上に置かれるため、意図的にマウント先を変えない限り問題ない。

---

### 2.8 内容を確認して起動する

右側の「概要」パネルで、AMI・インスタンスタイプ・キーペア・セキュリティグループ・
ストレージが意図どおりかを確認し、「**インスタンスを起動**」を押す。

**この時点から課金が始まる。** 1章のコスト表のとおり `g4dn.xlarge` は約109円/時である。
インスタンス一覧で「ステータスチェック」が **2/2 のチェックに合格** になるまで数分待つ。

---

### 2.9 SSHで接続する

```bash
ssh -i ~/.ssh/rag-poc.pem ubuntu@<EC2のパブリックIP>
```

初回接続時は接続先のフィンガープリントの確認を求められるので `yes` と答える。
ログインユーザー名は、Ubuntu系AMIなので `ubuntu` である。

#### 接続先を `~/.ssh/config` に登録しておく

2.10のとおりパブリックIPは停止・起動のたびに変わる。毎回コマンドに直接書いていると、
変わるたびにSSHとrsyncの両方を直すことになる。
`~/.ssh/config` に書いておけば、修正は `HostName` の1行で済む。

```bash
cat >> ~/.ssh/config <<'EOF'

Host rag-poc
    HostName <EC2のパブリックIP>
    User ubuntu
    IdentityFile ~/.ssh/rag-poc.pem
EOF
chmod 600 ~/.ssh/config
```

以降はこれだけで接続できる。

```bash
ssh rag-poc
```

インスタンスを停止・起動してIPが変わったら、`~/.ssh/config` の `HostName` の行だけを書き換える。

#### GPUの確認

接続できたら、GPUが見えていることを確認する。

```bash
nvidia-smi
```

GPU名（Tesla T4）とGPUメモリ量（約15360MiB）、および右上にCUDA Versionが表示されれば正常である。
ここまで通れば3章の下準備はほぼ完了しているので、3.1〜3.3は確認のみで通過できる。

#### 別のマシンを追加する（鍵を別途発行する）

2.5の `rag-poc.pem` は再取得できないため、別のMacやメンバーを追加するときに
これをコピーして配るのは避ける。追加するマシン側で新しい鍵を作り、
その**公開鍵だけ**をサーバに登録する。

```bash
# 追加するマシンで実行
ssh-keygen -t ed25519 -f ~/.ssh/rag-poc-<名前> -C "rag-poc <名前>"
```

パスフレーズは任意で、空のままEnterでもよい。
`~/.ssh/rag-poc-<名前>`（秘密鍵）と `~/.ssh/rag-poc-<名前>.pub`（公開鍵）ができる。

```bash
# 追加するマシンで公開鍵を表示し、1行まるごとコピーする
cat ~/.ssh/rag-poc-<名前>.pub
```

```bash
# 既に接続できるマシンから
ssh rag-poc

# EC2側で実行。<公開鍵の内容> は上でコピーした1行をそのまま貼る
echo '<公開鍵の内容>' >> ~/.ssh/authorized_keys
```

`~/.ssh` は 700、`authorized_keys` は 600 である必要がある。AMI既定のままなら変更は不要だが、
追記しても接続できない場合は `ls -ld ~/.ssh ~/.ssh/authorized_keys` で確認する。

追加したマシンからは、2.9と同じ要領で `~/.ssh/config` に書いて接続する。
`IdentityFile` だけが `~/.ssh/rag-poc-<名前>` になる。

**秘密鍵そのものを共有しないこと。** 誰がいつ入ったかを追えなくなり、
特定の1人だけアクセスを止めることもできなくなる。マシンごと・人ごとに鍵を分け、
不要になったら `authorized_keys` から該当行を削除する。

---

### 2.10 停止・再開するときの注意

1章のとおり、検証中は使わない時間帯にインスタンスを停止する運用を強く勧める。その際に2点注意する。

- **パブリックIPは停止・起動のたびに変わる。** SSHの接続先も、ブラウザで開く
  `http://<IP>:8501` も毎回変わる。固定したい場合はElastic IPを割り当てるが、
  **確保したIPv4アドレスは保有しているだけで課金対象**になる点に注意する
- **EBSの料金は停止中も発生し続ける。** 止まるのはインスタンス料金だけである
  （100GB gp3 でおよそ月1,500円）

---

## 3. サーバ側の下準備

> **2.3の Deep Learning Base OSS Nvidia Driver GPU AMI を選んだ場合、
> 3.1〜3.3はすべて導入済みである。** 各節の確認コマンドだけを実行して通過してよい。
> 通常のUbuntu AMIから始めた場合のみ、導入手順を実施する。
> 3.4（ソースコードの配置）はどちらの場合も必要である。

### 3.1 GPUドライバの確認

```bash
nvidia-smi
```

GPU名（Tesla T4）とメモリ量が表示されれば導入済み。`command not found` の場合はドライバが入っていないので、
2.3のDeep Learning Base OSS Nvidia Driver GPU AMIで作り直すか、NVIDIAドライバを導入する。

> **カーネルを不用意に更新しないこと。** AWSはこのAMIについて、
> 「セキュリティパッチによる場合を除き、カーネルバージョンを更新しないこと」を推奨している。
> 導入済みドライバとの互換性が壊れ、`nvidia-smi` が動かなくなることがある。
> `apt upgrade` を実行する際は、カーネルパッケージ（`linux-image-*` / `linux-aws`）が
> 対象に含まれていないかを確認する。

### 3.2 Docker と Docker Compose

**確認**：`docker --version` と `docker compose version` が通れば導入済みで、本節は不要である
（Deep Learning Base AMIには同梱されている）。

```bash
# Dockerの導入
curl -fsSL https://get.docker.com | sudo sh

# sudoなしでdockerを使えるようにする（実行後、一度ログアウトして入り直す）
sudo usermod -aG docker $USER
```

### 3.3 NVIDIA Container Toolkit（重要）

**これが無いとコンテナからGPUが見えない。** ホストにドライバがあるだけでは不十分である。

**確認**：`nvidia-ctk --version` が通れば導入済みで、以下の導入手順は不要である
（Deep Learning Base AMIには同梱されている）。
**ただし、末尾の動作確認コマンドは必ず実行すること。**

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

リポジトリのリモートはSSH形式（`git@github.com:<ユーザー名>/<リポジトリ名>.git`）である。
EC2からクローンするには、**EC2用の鍵をGitHubにデプロイキーとして登録する**必要がある。
2.5で作った `rag-poc.pem` はEC2にログインするための鍵であり、GitHubとは無関係である。

#### EC2側で鍵を作る

```bash
# EC2上で実行
ssh-keygen -t ed25519 -f ~/.ssh/github-deploy -C "rag-poc ec2 deploy key" -N ""
```

`-N ""` はパスフレーズを空にする指定である。パスフレーズを付けると `git pull` のたびに入力を求められる。

```bash
cat ~/.ssh/github-deploy.pub
```

#### GitHubに登録する

ブラウザで対象リポジトリを開き、**Settings → Deploy keys → Add deploy key** で登録する。

| 項目 | 値 |
|---|---|
| Title | `rag-poc EC2` |
| Key | 上で表示した公開鍵の1行をそのまま貼る |
| Allow write access | **チェックしない**（読み取り専用） |

デプロイキーはリポジトリ単位の鍵で、アカウント全体の権限を持たない。
サーバが侵害された場合の影響を、このリポジトリの読み取りだけに限定できる。
EC2からコードを書き戻すことはないため、書き込みは許可しない。

> **デプロイキーはリポジトリの公開・非公開にかかわらず使える。**
> publicリポジトリならHTTPS URLで認証なしにクローンすることもできるが、
> privateに変更した時点で動かなくなる。デプロイキーで統一しておけばどちらでも同じ手順になる。

#### GitHubにこの鍵を使わせる

```bash
# EC2上で実行
cat >> ~/.ssh/config <<'EOF'

Host github.com
    User git
    IdentityFile ~/.ssh/github-deploy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

登録できたか確認する。

```bash
ssh -T git@github.com
```

初回はフィンガープリントの確認を求められるので `yes` と答える。
`Hi <ユーザー名>/<リポジトリ名>! You've successfully authenticated, but GitHub does not provide shell access.`
と表示されれば成功である。

#### クローンする

```bash
git clone <リポジトリのURL> ~/rag-poc
cd ~/rag-poc
```

6章の `git pull` も、この設定のまま追加の操作なしで通る。

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

### 4.3 【用意済み】`pyproject.toml` ― PyTorchは環境に応じて自動で切り替わる

**手作業での差し替えは不要である。** 実行環境に応じて自動で選択される。

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu126", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
  { index = "pytorch-cpu",   marker = "sys_platform != 'linux' or platform_machine != 'x86_64'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

markerにより、次のように振り分けられる。

| 環境 | 判定 | 入るPyTorch |
|---|---|---|
| EC2（`g4dn.xlarge`） | linux / x86_64 | `torch 2.13.0+cu126`（CUDA版） |
| MacのDevコンテナ（Apple Silicon） | linux / aarch64 | `torch 2.13.0+cpu` |
| macOSホスト | darwin | `torch 2.13.0` |

`uv.lock` には3系統すべてが記録済みである。したがって **EC2側で `pyproject.toml` を編集する必要も、
`uv lock` を実行する必要もない**（uvをサーバに導入する必要もない）。

> **CUDAを `cu126` にしている理由**：
> 1. `torch 2.13.0` のwheelは `cu124` には配布されていない（cu124の最新は 2.6.0 まで）
> 2. Deep Learning Base OSS Nvidia Driver GPU AMI に同梱されるCUDAが 12.6 である（2.3）
>
> torchのバージョンを変更する際は、`https://download.pytorch.org/whl/cu126/torch/` に
> 該当バージョンのwheelが存在するかを必ず確認すること。

> **注意**：CUDA版のPyTorchはCPU版より2GB以上大きい。EC2でのイメージビルド時間とEBS消費が増える。
>
> **Ollama（LLM推論）はこの設定と無関係にGPUを使う。** Ollamaは自前のランタイムを持つため、
> この設定が影響するのは Embedding（BGE-M3）だけである。

### 4.4 【用意済み】`src/infrastructure/llama_index_factory.py`

Embeddingを動かすデバイスは、環境変数 `EMBEDDING_DEVICE` で切り替わるようにしてある
（既定は `cpu`）。`docker-compose.gpu.yml` がこれを `cuda` に設定するため、
GPU環境ではコードを変更する必要はない。

```python
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
USE_FP16 = EMBEDDING_DEVICE.startswith("cuda")  # fp16はGPUでのみ有効
```

`use_fp16` はCPUでは無効（むしろ遅くなる）でGPUでのみ有効なため、デバイスに連動させている。

4.3のとおりEC2（linux/x86_64）では自動的にCUDA版のPyTorchが入るため、この設定はそのまま機能する。
CPU専用版のPyTorchはCUDAデバイスを認識できないが、その組み合わせはmarkerにより発生しない。

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
cd ~/rag-poc

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

**`docs/` はリポジトリに含まれない。** `.gitignore` で除外されているため（社内文書のため）、
`git clone` してもEC2側は空である。Mac側から直接転送する。

```bash
# Mac側で実行（2.9で ~/.ssh/config に rag-poc を登録してあること）
rsync -avz --exclude '*.pem' --exclude '.DS_Store' \
  ./docs/ rag-poc:~/rag-poc/docs/
```

転送先は `docker-compose.prod.yml` が読み取り専用でマウントするディレクトリである。
ドキュメントを差し替えた際は、同じコマンドで再転送してから取り込みを実行する
（イメージの作り直しは不要）。

```bash
# 以下はEC2側で実行
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
| 起動時に `VcpuLimitExceeded` で失敗する | GPUインスタンスのvCPUクォータが不足。2.1で確認・申請する |
| 起動時に `InsufficientInstanceCapacity` で失敗する | そのAZにGPUインスタンスの空きが無い。別のAZのサブネットを選んで再試行する（2.4） |
| SSHで `UNPROTECTED PRIVATE KEY FILE!` と出て接続できない | `.pem` の権限が緩い。`chmod 400` する（2.5） |
| 昨日つながっていたIPにSSHもブラウザも到達しない | インスタンスを停止・起動してパブリックIPが変わった。現在のIPを確認する（2.10）。作業場所のグローバルIPが変わった場合はセキュリティグループの更新も必要（2.6） |
| 再起動後に `nvidia-smi` が `command not found` / エラーになる | カーネルが更新されてドライバと不整合を起こした可能性がある（3.1の注記） |
| `docker: Error response ... could not select device driver "nvidia"` | NVIDIA Container Toolkit が未導入。3.3を実施する |
| 回答生成が異常に遅い（1分以上） | GPUが使われていない。`nvidia-smi` でollamaプロセスを確認。5.3を参照 |
| `Connection refused` でOllamaにつながらない | `docker-compose.override.yml` を本番に持ち込んでいないか確認（4.5） |
| 質問しても必ず「関連する文書が見つかりません」 | ドキュメント未取り込み。5.4を実施。5.5でチャンク数が0でないか確認 |
| ブラウザから8501につながらない | セキュリティグループで8501が許可されているか、`--server.address 0.0.0.0` が付いているか確認 |
| メモリ不足でコンテナが落ちる | BGE-M3がdense用・sparse用の2つ常駐し約4.5GB使う。インスタンスのRAMを確認する |
