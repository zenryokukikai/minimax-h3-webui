# minimax-h3-webui

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — 動画とネイティブステレオ音声を
単一の forward で同時生成する 33B のオムニモーダルモデル — を、ローカルGPUで動かすための
**簡易 Web フロントエンド**と、**GPU世代ごとの量子化選択ガイド**です。

ノードグラフではなく、プロンプト・解像度・秒数・参照画像だけのシンプルな画面を提供します。
生成エンジンには ComfyUI をヘッドレスで使い、その HTTP API を薄くラップしています。

<!-- 画面: モード切替 / プロンプト / 解像度 / 長さ / シード / 進捗 / 結果プレビュー -->

## なぜこのリポジトリがあるか

MiniMax-H3 を量子化して動かす手順は各所にありますが、**推奨される量子化形式が GPU 世代で
真逆になる**ことがあまり書かれていません。コミュニティの既定値をそのまま持ってくると、
自分のGPUでは動かないか、エミュレーションで数倍遅くなります。

実機で測って整理した結果が以下です。

## GPU世代ごとの量子化選択

| GPU世代 | 例 | 選ぶべき形式 | 理由 |
|---|---|---|---|
| **Blackwell** (sm_120/121) | RTX 50xx, GB10, B200 | **NVFP4** | 第5世代 Tensor Core が FP4 をネイティブ実行。ファイルサイズが約半分 |
| **Ada** (sm_89) | RTX 40xx, L40S | FP8 scaled または INT8 ConvRot | FP8 がネイティブ |
| **Ampere** (sm_80/86) | A100, RTX 30xx | **INT8 ConvRot** | FP4/FP8 のハードウェア命令がない。INT8 Tensor Core は強力 |

ComfyUI の起動ログで実際にどちらになっているか確認できます。Ampere での例:

```
Native ops:   convrot_w4a4, int8_tensorwise
emulated ops: float8_e4m3fn, nvfp4, float8_e5m2, mxfp8
```

`emulated` に入っている形式を選ぶと、ファイルは小さくなっても速度は落ちます。

### さらに注意点

- **NVFP4 を選んでも速くはならない。** Blackwell で NVFP4 と INT8 を比較した公開ベンチでは
  差は数%で、しかもクロック差に起因すると分析されています。video diffusion は
  **compute-bound であって bandwidth-bound ではない**ため、量子化形式より演算スループットが効きます。
  NVFP4 の価値はファイルサイズとメモリ使用量です。
- **comfy-kitchen の CUDA バックエンドは PyTorch cu130 以上を要求します。**
  ドライバが CUDA 13 に届かない環境（CUDA 13 にはドライバ 580 以上が必要）では
  無効化され、eager 実行になって遅くなります。その場合は
  `--enable-triton-backend` で Triton 実装のカーネルを使ってください。
  データセンタGPUなら `cuda-compat` による forward compatibility も選択肢です。
- **`pruned` は情報を捨てていません。** 33B のうち AdaLN 分岐の 13B は推論時に
  事前計算できるため、それを畳んだものです（BF16 で 66.3GB → 40.2GB）。推論専用なら
  pruned で問題ありません。

## 必要なモデルファイル

[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) が公式の再パッケージ版です。
NVFP4 の DiT だけは別リポジトリ（例: [Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot)）にあります。

| 役割 | 置き場所 | Ampere 向け | Blackwell 向け |
|---|---|---|---|
| DiT（T2V/I2V/先頭末尾フレーム） | `models/diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot` (21.0GB) | `MiniMax_H3_FL2VA_pruned_nvfp4` (12.5GB) |
| DiT（参照画像） | `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot` (21.0GB) | `MiniMax_H3_Ref2VA_pruned_nvfp4` (12.5GB) |
| テキストエンコーダ (Qwen3-VL-32B) | `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot` (27.1GB) | `qwen3vl_32b_minimax_h3_nvfp4_awq` (15.7GB) |
| Video VAE | `models/vae/` | `minimax_h3_video_vae_fp16` (5.2GB) | 同左 |
| Audio VAE | `models/vae/` | `minimax_h3_audio_vae_fp32` (0.6GB) | 同左 |

ファイル名は `app/models.json` で差し替えられます（`app/models.json.example` 参照）。

## セットアップ

ComfyUI 0.30.0 以降が必要です（MiniMax-H3 のネイティブ対応が入ったバージョン）。

```bash
git clone https://github.com/zenryokukikai/minimax-h3-webui.git
cd minimax-h3-webui

# 1. ComfyUI と Python 環境
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
uv venv --python 3.12 venv
uv pip install --python venv/bin/python --torch-backend=cu130 torch torchvision torchaudio
uv pip install --python venv/bin/python -r ComfyUI/requirements.txt
```

`--torch-backend` はドライバに合わせてください（CUDA 12.x なら `cu128` など）。

```bash
# 2. ComfyUI にモデル置き場を教える
cat > ComfyUI/extra_model_paths.yaml <<'EOF'
minimax_h3:
    base_path: /path/to/minimax-h3-webui/models
    diffusion_models: diffusion_models
    text_encoders: text_encoders
    vae: vae
EOF

# 3. 設定
cp scripts/config.example.sh scripts/config.sh
$EDITOR scripts/config.sh

# 4. 重みを models/ 以下に配置してから起動
bash scripts/start.sh
```

停止は `bash scripts/stop.sh`。別マシンへ配置するなら `bash scripts/deploy.sh`。

ComfyUI は常に `127.0.0.1` で待ち受け、外部に出るのは Web UI だけです。
ノードエディタを直接触りたい場合は SSH ポートフォワードしてください。

```bash
ssh -L 18188:127.0.0.1:18188 your-host
```

## GPU別の実測

864x480 / 20ステップ / 124フレーム（5.17秒）/ 同一プロンプト・同一シード。
それぞれのGPUで**最適な**量子化形式とバックエンドを使った場合の値です。

| GPU | 量子化 | comfy-kitchen バックエンド | s/it | 合計 |
|---|---|---|---|---|
| A100 80GB (sm_80) | INT8 ConvRot | Triton（ドライバが CUDA 13 未満のため） | 4.25 | 110秒 |
| GB10 / DGX Spark (sm_121) | NVFP4 | CUDA（cu130 ネイティブ） | 7.87 | 203秒 |

**A100 が約1.85倍速い**という結果でした。GB10 は 128GB 統合メモリで
大きなモデルが載る点が強みであって、単機の生成速度で A100 に勝つ構成では
ありません（LPDDR5X は容量に対して帯域が低い）。ただし GB10 は
NVFP4 がネイティブで動くぶんファイルサイズが約半分（46GB 対 74GB）で済みます。

なお公開されている DGX Spark のベンチではもっと遅い数字（5秒480pで6分程度）も
報告されていますが、それらは量子化形式やバックエンドが最適でない構成と思われます。

## API

Web UI と同じ `/v1` を、そのままプログラムから呼べます（UI 自身がこの API の
クライアントです）。機械可読な定義は `GET /v1/openapi.json`。

### 投入 → 完了検知 → ダウンロード

**投入は即座に id を返します**（完了は待ちません）。

```bash
curl -sX POST http://your-host:18190/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"雨の夜の路地。音: 雨音と遠くの雷","width":864,"height":480,"seconds":5}'
```

```json
{
  "id": "9e23c38f-6e37-4438-af24-89b88c8e90b2",
  "status": "queued",
  "queue_position": 1,
  "progress": {"step": 0, "total": 20, "percent": 0.0},
  "request": {"seed": 4821..., "width": 864, "height": 480, "seconds": 5.17, "length": 124},
  "relative_cost": 1.0
}
```

状態を見て、`status` が `done` になったら `video_url` が入ります。

```bash
curl -s http://your-host:18190/v1/jobs/$ID
```

```json
{
  "id": "9e23c38f-...", "status": "done",
  "progress": {"step": 20, "total": 20, "percent": 100.0},
  "elapsed_sec": 110.2,
  "video_url": "/v1/jobs/9e23c38f-.../video"
}
```

```bash
curl -o out.mp4 "http://your-host:18190/v1/jobs/$ID/video"
```

`status` は `queued` → `running` → `decoding` → `done` と遷移します
（異常時は `error` / `cancelled`）。`queue_position` は 0 が実行中、
1 以上が「自分より前に残っている件数」です。

短いクリップなら `"wait": true` を付けると完了までブロックして
最終結果をそのまま返します（`"timeout"` 秒で打ち切り、既定1800秒）。

### 複数リクエストのキューイング

ComfyUI は一度に1件しか実行しないため、**続けて投入すれば自動的に直列のキューに並びます**。
投入側は待つ必要がなく、id は数ミリ秒で返ります。

```bash
curl -s http://your-host:18190/v1/queue
```

```json
{"running": [{"id": "...", "queue_position": 0, "status": "running", ...}],
 "pending": [{"id": "...", "queue_position": 1, ...}],
 "pending_count": 1}
```

取り消しは対象がどこにいるかで動作が変わります。**キュー待ちならその1件だけを
キューから外し、実行中なら中断します**（待機中のジョブを取り消しても、
実行中の別ジョブは巻き込まれません）。

```bash
curl -sX POST http://your-host:18190/v1/jobs/$ID/cancel
```

### エンドポイント

| メソッド | パス | 用途 |
|---|---|---|
| `GET` | `/v1/options` | 解像度プリセット・サンプラー・コスト式・モデル配置状況 |
| `POST` | `/v1/images` | 参照画像を登録して `ref` を得る（multipart の `file`、または JSON の `data` に base64） |
| `POST` | `/v1/generate` | 生成を投入。既定は非同期で即 id を返す |
| `GET` | `/v1/jobs` | 最近のジョブ一覧（`?limit=`） |
| `GET` | `/v1/jobs/{id}` | 状態・進捗・完了時の `video_url` |
| `GET` | `/v1/jobs/{id}/video` | mp4（音声つき）。`?download=1` で添付ダウンロード |
| `POST` | `/v1/jobs/{id}/cancel` | キューから外す、または実行中なら中断 |
| `GET` | `/v1/queue` | キューの現況 |
| `GET` | `/healthz` | ヘルスチェック |

### `POST /v1/generate` のパラメータ

| キー | 既定 | 説明 |
|---|---|---|
| `prompt` | 必須 | 映像と音の内容。音の指示もここに書く |
| `mode` | `t2v` | `t2v` / `i2v` / `ref2v` |
| `width` `height` | 864 / 480 | 32の倍数に丸められる |
| `seconds` | 5 | 24fps の 17k+5 フレーム格子に切り上げ |
| `steps` | 20 | |
| `seed` | ランダム | 省略・`null`・`-1` でランダム |
| `sampler` `scheduler` | `res_multistep` / `simple` | |
| `first_frame` `last_frame` | — | `mode=i2v`。`/v1/images` が返す `ref` |
| `ref_images` | — | `mode=ref2v`。最大9枚。プロンプト中で `<Picture 1>` … と参照 |
| `ref_image_size` | `match` | `max` は同一性重視だが数倍遅い |
| `wait` `timeout` | `false` / 1800 | `true` で完了までブロック |

エラーは HTTP ステータスと `{"error": {"code", "message"}}` で返ります。

### 認証

`H3_API_KEY` を設定すると `/v1/*` に `Authorization: Bearer <key>` が必要になります
（`X-API-Key` ヘッダでも可）。未設定なら認証なしです。Web UI は 401 を受けると
キーの入力を求め、`localStorage` に保存します。

### Python クライアント

`tools/h3_client.py` が標準ライブラリだけで動くクライアントです。

```python
from h3_client import H3Client

c = H3Client("http://your-host:18190")       # 既定は $H3_URL
job = c.generate("雨の夜の路地。音: 雨音", width=864, height=480)   # 即 id が返る
final = c.wait(job["id"], on_progress=lambda s: print(s["status"], s["progress"]))
c.download(job["id"], "out.mp4")
```

CLI としても使えます。

```bash
export H3_URL=http://your-host:18190
python tools/h3_client.py options
python tools/h3_client.py generate "雨の夜の路地。音: 雨音" -o out.mp4
python tools/h3_client.py generate "..." --mode i2v --image first.png -o out.mp4
python tools/h3_client.py jobs
```

`tools/test_queue.py` はキューイングの結合テストです（複数投入・順位・
待機中だけの取消・完了検知・ダウンロードまでを通しで確認します）。

## 生成コストは画素数でほぼ決まる

A100 80GB / INT8 ConvRot / 20ステップ / 124フレーム での 1ステップあたり実測値:

| 画素数 | s/it | 画素あたり |
|---|---|---|
| 0.068 MP (352x192) | 0.91 | 74.7k px/s |
| 0.115 MP (448x256) | 1.40 | 82.1k px/s |
| 0.184 MP (576x320) | 2.10 | 87.6k px/s |
| 0.258 MP (672x384) | 3.00 | 86.0k px/s |
| 0.344 MP (768x448) | 4.00 | 86.0k px/s |
| 0.737 MP (1152x640) | 10.0 | 73.7k px/s |
| 1.032 MP (1344x768) | 16.9 | 61.1k px/s |

小さいうちはほぼ線形（固定オーバーヘッド分だけ効率が落ちる）、大きくなると
アテンションの O(n²) が効いて超線形になります。これは

```
cost(px) ≈ 0.332 + 7.968·MP + 7.836·MP²
```

でおおむね5%以内に収まります。UI の解像度リストにはこの式から出した
**864x480 を 1.00 とした相対コスト**を表示しています。絶対時間は GPU で桁が変わるので出しません
（同じ設定でも A100 80GB と DGX Spark で3倍以上違います）。

## 解像度と画質の下限

同一プロンプト・同一シードで中間フレームを目視比較した結果:

| 解像度 | 相対コスト | 画質 |
|---|---|---|
| 1344x768（ネイティブ） | 3.39× | 最高 |
| 864x480 | 1.00× | 良好 |
| 672x384 | 0.58× | 良好（短辺384px＝公式の下限） |
| **576x320** | 0.41× | 良好（**実用下限**） |
| 448x256 / 384x256 | 0.27× / 0.24× | ⚠ 半透明のゴースト・二重像が出る |
| 352x192 | 0.18× | ⚠ 傘や人物のシルエットが崩れる |

ノードは32の倍数なら受け付け、352x192 でも**エラーなく完走します**。ただし完走＝破綻していない、
ではありません。352x192（0.18×）で構図とシードの当たりを高速に探し、決まったら
本番解像度で回すのが効率的です。同じシードなら構図の大枠は保たれます。

`tools/sweep_resolution.py` で自分の環境の同じ表が作れます。

## 使い方

- **テキスト→動画** / **画像→動画**（先頭・末尾フレーム指定）/ **参照画像**（最大9枚）
- 長さは 5〜15秒。24fps の 17k+5 フレーム格子に自動でスナップします（学習範囲は約5〜15秒）
- 音声は映像と同時に生成されるので、**プロンプトに音の指示も書いてください**

```
夕暮れの東京の路地。ネオンが濡れた路面に映り込む。手持ちカメラでゆっくり前進。
[0s-2s] 傘をさした人物の背中を追う。
[2s-5s] 振り返って正面のクローズアップ、雨音が強まる。
音: 雨音、遠くの electronic な BGM、2秒地点で雷鳴。
```

参照画像モードではプロンプト中で `<Picture 1>`, `<Picture 2>` … と参照します。

公式のプロンプトガイドが
[base](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md) /
[reference](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
にあります。

## 構成

```
app/
  server.py          aiohttp。ComfyUI の API を仲介し、進捗を WebSocket で購読
  workflows.py       ComfyUI の API 形式ワークフローの組み立てと解像度プリセット
  static/index.html  画面（依存ライブラリなしの単一ファイル）
  models.json        モデルファイル名の上書き（任意、gitignore 済み）
scripts/
  start.sh  stop.sh  deploy.sh  config.example.sh
tools/
  h3_client.py         API の Python クライアント兼 CLI（標準ライブラリのみ）
  sweep_resolution.py  解像度ごとのコストと画質を実測する
  test_queue.py        キューイングの結合テスト
```

公式ワークフローテンプレート (`video_minimax_h3_{t2v,i2v,r2v}.json`) のサブグラフを
API 形式に展開したものが `workflows.py` です。ノード入力名は ComfyUI の
`/api/object_info` と `comfy_extras/nodes_minimax_h3.py` で確認しています。

## 既知の制約

- 256p 未満など極端に小さい解像度は品質が破綻します（上表参照）
- ComfyUI 0.30.x には pinned memory の退行があり、`--disable-pinned-memory` を既定で付けています
- 2K 出力には別モジュール（H3-Regenerate-2K）が必要で、オープンウェイトには含まれません
- テキストエンコーダは `lm_head` などが削られているため、テキスト生成用途には使えません

## ライセンス

このリポジトリのコードは MIT ライセンスです。**モデルの重みは対象外**で、
[MiniMax-H3 のライセンス](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
に従ってください。
