"""解像度ごとの生成コストと画質を実測する。

同一プロンプト・同一シードで解像度だけを変えて回し、所要時間を出す。
画質は自動判定できないので、出力された動画を目視で比較すること
（低解像度では「エラーなく完走するが破綻している」という状態になる）。

使い方:
    python tools/sweep_resolution.py                       # 既定の 16:9 ラダー
    python tools/sweep_resolution.py 1344x768 864x480      # 解像度を指定
    H3_UI=http://127.0.0.1:18190 python tools/sweep_resolution.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

UI = os.environ.get("H3_UI", "http://127.0.0.1:18190").rstrip("/")
STEPS = int(os.environ.get("H3_STEPS", "20"))
SECONDS = float(os.environ.get("H3_SECONDS", "5"))
SEED = int(os.environ.get("H3_SEED", "12345"))

PROMPT = os.environ.get("H3_PROMPT") or (
    "Realistic handheld shot: a rainy neon alley at night, reflections on wet asphalt, "
    "slow forward dolly. A person with a transparent umbrella walks away from camera, "
    "then turns back to face the lens. Audio: heavy rain, distant traffic, a low synth "
    "pad, one thunder crack at 3s. No text, no watermark."
)

DEFAULT_CASES = ["1344x768", "864x480", "768x448", "672x384",
                 "576x320", "448x256", "384x256", "352x192"]


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        UI + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def run(width: int, height: int) -> dict:
    t0 = time.time()
    job = _post("/api/generate", {
        "mode": "t2v", "prompt": PROMPT,
        "width": width, "height": height,
        "seconds": SECONDS, "steps": STEPS, "seed": SEED,
    })
    pid = job["prompt_id"]
    while time.time() - t0 < 3600:
        s = json.load(urllib.request.urlopen(f"{UI}/api/status/{pid}"))
        if s.get("state") == "done":
            elapsed = time.time() - t0
            return {"ok": True, "sec": round(elapsed, 1),
                    "s_per_it": round(elapsed / STEPS, 2),
                    "video": s["videos"][0]}
        if s.get("state") == "error":
            return {"ok": False, "error": (s.get("error") or "")[:300]}
        time.sleep(2)
    return {"ok": False, "error": "timeout"}


def main() -> int:
    cases = sys.argv[1:] or DEFAULT_CASES
    results = []
    for case in cases:
        try:
            w, h = (int(v) for v in case.lower().split("x"))
        except ValueError:
            print(f"解像度の書式が不正です: {case}（例: 864x480）", file=sys.stderr)
            return 2
        r = {"res": case, "px": w * h, **run(w, h)}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    print("\n=== まとめ ===")
    base = next((r for r in results if r.get("ok")), None)
    for r in results:
        if not r.get("ok"):
            print(f"{r['res']:>10}  {r['px'] / 1000:6.0f}k px  FAILED: {r.get('error')}")
            continue
        rel = f"{r['sec'] / base['sec']:5.2f}×" if base else "     "
        print(f"{r['res']:>10}  {r['px'] / 1000:6.0f}k px  {r['sec']:7.1f}s  "
              f"{r['s_per_it']:5.2f} s/it  相対 {rel}  {r['video']}")
    print("\n※ 完走しても画質が破綻していることがあります。出力を目視で確認してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
