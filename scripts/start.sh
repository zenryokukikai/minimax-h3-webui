#!/usr/bin/env bash
# ComfyUI（生成エンジン）と Web フロントエンドを起動する。
# 設定は scripts/config.sh か環境変数で与える（scripts/config.example.sh 参照）。
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/config.sh" ] && . "$HERE/config.sh"

ROOT=${H3_ROOT:-$(cd "$HERE/.." && pwd)}
COMFY_PORT=${H3_COMFY_PORT:-18188}
UI_PORT=${H3_UI_PORT:-18190}
UI_HOST=${H3_UI_HOST:-127.0.0.1}
EXTRA=${H3_COMFY_EXTRA_ARGS:-}
PY=$ROOT/venv/bin/python

mkdir -p "$ROOT/logs"

start_comfy() {
  if pgrep -f "ComfyUI/main.py .*--port $COMFY_PORT" > /dev/null; then
    echo "ComfyUI は既に起動しています (port $COMFY_PORT)"
    return
  fi
  echo "ComfyUI を起動します…"
  # --disable-pinned-memory: ComfyUI 0.30.x の pinned memory 退行による
  #   ロード遅延を回避する。
  # 追加フラグ (H3_COMFY_EXTRA_ARGS) は GPU 世代に応じて config.sh で指定。
  # shellcheck disable=SC2086
  setsid nohup "$PY" "$ROOT/ComfyUI/main.py" \
    --listen 127.0.0.1 --port "$COMFY_PORT" \
    --disable-pinned-memory $EXTRA \
    > "$ROOT/logs/comfyui.log" 2>&1 < /dev/null &
  echo "  PID $! / ログ: $ROOT/logs/comfyui.log"
}

start_ui() {
  if pgrep -f "app/server.py .*--port $UI_PORT" > /dev/null; then
    echo "Web UI は既に起動しています (port $UI_PORT)"
    return
  fi
  echo "Web UI を起動します…"
  setsid nohup "$PY" "$ROOT/app/server.py" \
    --host "$UI_HOST" --port "$UI_PORT" \
    --comfy "http://127.0.0.1:$COMFY_PORT" \
    > "$ROOT/logs/ui.log" 2>&1 < /dev/null &
  echo "  PID $! / ログ: $ROOT/logs/ui.log"
}

start_comfy
sleep 3
start_ui
echo
echo "  Web UI : http://$UI_HOST:$UI_PORT"
echo "  ComfyUI: http://127.0.0.1:$COMFY_PORT （ノードエディタは SSH ポートフォワード経由）"
