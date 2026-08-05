#!/usr/bin/env bash
# ローカルの app/ scripts/ tools/ をリモートホストへ同期する。
# 接続先は scripts/config.sh か環境変数で指定（scripts/config.example.sh 参照）。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/config.sh" ] && . "$HERE/config.sh"

HOST=${H3_SSH_HOST:?H3_SSH_HOST が未設定です（scripts/config.sh を作成してください）}
ROOT=${H3_ROOT:?H3_ROOT が未設定です}
LOCAL=$(cd "$HERE/.." && pwd)

rsync -a --delete --exclude "__pycache__" --exclude "models.json" \
  "$LOCAL/app/" "$HOST:$ROOT/app/"
rsync -a --exclude "config.sh" "$LOCAL/scripts/" "$HOST:$ROOT/scripts/"
rsync -a "$LOCAL/tools/" "$HOST:$ROOT/tools/"
ssh "$HOST" "chmod +x $ROOT/scripts/*.sh"
echo "同期完了: $HOST:$ROOT"
