# scripts/config.sh としてコピーし、環境に合わせて書き換えてください。
# config.sh は .gitignore 済みです（ホスト名・IP を含むため）。
#
# すべて `:=` 形式なので、環境変数を先に与えればそちらが優先されます。
#   H3_SSH_HOST=user@other-host bash scripts/deploy.sh

# デプロイ先（ローカルで動かすだけなら不要）
: "${H3_SSH_HOST:=user@your-host}"

# 設置ディレクトリ
: "${H3_ROOT:=/opt/minimax-h3}"

# Web UI の待ち受けアドレス。
# 既定は 127.0.0.1（外部非公開）。LAN や VPN 上の他マシンから開くなら
# そのインターフェースのアドレスを指定してください。0.0.0.0 は到達可能な
# 全ネットワークに公開されるので、意図する場合のみ。
: "${H3_UI_HOST:=127.0.0.1}"
: "${H3_UI_PORT:=18190}"

# ComfyUI は常に 127.0.0.1 で待ち受けます（外部に晒さない）
: "${H3_COMFY_PORT:=18188}"

# 設定すると /v1/* に Authorization: Bearer <key> が必要になります。
# 空なら認証なし。ローカルホスト以外に公開するなら設定を推奨。
: "${H3_API_KEY:=}"

# GPU 世代に応じた追加フラグ
#   Ampere (A100 / RTX 30xx) でドライバが CUDA 13 に届かない場合、
#   comfy-kitchen の CUDA バックエンドが無効になるため Triton 版を使う:
#     : "${H3_COMFY_EXTRA_ARGS:=--enable-triton-backend}"
#   Blackwell (GB10 / RTX 50xx) で cu130 が載っているなら指定不要。
: "${H3_COMFY_EXTRA_ARGS:=}"

export H3_SSH_HOST H3_ROOT H3_UI_HOST H3_UI_PORT H3_COMFY_PORT H3_COMFY_EXTRA_ARGS H3_API_KEY
