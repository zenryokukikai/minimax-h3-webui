#!/usr/bin/env bash
# start.sh で起動したプロセスだけを停止する。
# PID を特定して個別に kill する（pkill -f は無関係なプロセスを巻き込むため使わない）。
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/config.sh" ] && . "$HERE/config.sh"

COMFY_PORT=${H3_COMFY_PORT:-18188}
UI_PORT=${H3_UI_PORT:-18190}

stop_one() {
  local pattern="$1" label="$2" pid
  pid=$(pgrep -f "$pattern" | head -1)
  if [ -z "$pid" ]; then
    echo "$label は起動していません"
    return
  fi
  echo "$label を停止します (PID $pid)"
  kill "$pid"
}

stop_one "app/server.py .*--port $UI_PORT" "Web UI"
stop_one "ComfyUI/main.py .*--port $COMFY_PORT" "ComfyUI"
