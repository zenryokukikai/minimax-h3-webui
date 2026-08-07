#!/usr/bin/env bash
# Range リクエストを並列に投げて大きなファイルを取得する。
#
# Hugging Face は未認証リクエストの単一接続を数 MB/s に絞ることがあり、
# 16GB 級のチェックポイントだと数時間かかってしまう。接続を分けると
# 合計帯域が出るため、分割して並列に取得してから連結する。
# 途中まで取れている場合はその続きから再開する。
#
#   usage: pdownload.sh <url> <dest> [並列数]
set -euo pipefail

URL=$1
DEST=$2
N=${3:-8}

SIZE=$(curl -sSLI "$URL" | tr -d '\r' | awk 'tolower($1) == "content-length:" { v = $2 } END { print v }')
if [ -z "$SIZE" ] || [ "$SIZE" -le 0 ] 2>/dev/null; then
  echo "サイズを取得できませんでした: $URL" >&2
  exit 1
fi

if [ -f "$DEST" ] && [ "$(stat -c%s "$DEST")" = "$SIZE" ]; then
  echo "取得済み: $DEST"
  exit 0
fi

echo "size=$SIZE parallel=$N -> $DEST"
mkdir -p "$(dirname "$DEST")" "$DEST.parts"
CHUNK=$(( (SIZE + N - 1) / N ))

for i in $(seq 0 $((N - 1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 ))
  [ "$END" -ge "$SIZE" ] && END=$(( SIZE - 1 ))
  [ "$START" -gt "$END" ] && continue

  PART="$DEST.parts/p$(printf '%03d' "$i")"
  HAVE=0
  [ -f "$PART" ] && HAVE=$(stat -c%s "$PART")
  WANT=$(( END - START + 1 ))
  [ "$HAVE" -ge "$WANT" ] && continue

  curl -sSL --retry 5 --retry-delay 3 -r "$(( START + HAVE ))-$END" "$URL" >> "$PART" &
done
wait

cat "$DEST.parts"/p* > "$DEST"
GOT=$(stat -c%s "$DEST")
if [ "$GOT" != "$SIZE" ]; then
  echo "サイズ不一致: $GOT != $SIZE（再実行すれば続きから取得します）" >&2
  exit 1
fi
rm -rf "$DEST.parts"
echo "OK $DEST ($GOT bytes)"
