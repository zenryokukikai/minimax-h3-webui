"""ステップ数を変えて、所要時間と出力の変化を実測する。

MiniMax-H3 の公式テンプレートは 20 ステップが既定。ステップ数の上限は
ComfyUI の BasicScheduler が 10000 まで受け付けるが、それは「モデルが
そこまで有効に使える」という意味ではない。実際にどこで頭打ちになるかを
同一プロンプト・同一シードで確認する。

    H3_URL=http://host:18190 python tools/sweep_steps.py
    H3_URL=... python tools/sweep_steps.py 10 20 40 80 --width 576 --height 320
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_client import H3Client, H3Error  # noqa: E402

PROMPT = os.environ.get("H3_PROMPT") or (
    "Realistic handheld shot: a rainy neon alley at night, reflections on wet asphalt, "
    "slow forward dolly. A person with a transparent umbrella walks away from camera, "
    "then turns back to face the lens. Audio: heavy rain, distant traffic, a low synth "
    "pad, one thunder crack at 3s. No text, no watermark."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", type=int, default=[10, 20, 40, 80])
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default="/tmp/h3_steps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    client = H3Client()
    rows = []

    for n in args.steps:
        try:
            job = client.generate(PROMPT, width=args.width, height=args.height,
                                  seconds=args.seconds, steps=n, seed=args.seed)
            final = client.wait(job["id"], timeout=3600)
        except H3Error as e:
            print(f"{n:>5} steps  FAILED: {e}", flush=True)
            rows.append((n, None, None))
            continue
        if final["status"] != "done":
            print(f"{n:>5} steps  FAILED: {final.get('error', final['status'])}", flush=True)
            rows.append((n, None, None))
            continue
        path = client.download(job["id"], f"{args.outdir}/steps_{n:03d}.mp4")
        sec = final["elapsed_sec"]
        rows.append((n, sec, path))
        print(f"{n:>5} steps  {sec:7.1f}s  {sec / n:5.2f} s/step  {path}", flush=True)

    print("\n=== まとめ ===")
    base = next((r for r in rows if r[1]), None)
    for n, sec, path in rows:
        if sec is None:
            print(f"{n:>5} steps  FAILED")
            continue
        rel = f"{sec / base[1]:5.2f}×" if base else ""
        print(f"{n:>5} steps  {sec:7.1f}s  相対 {rel}  {path}")
    print("\n※ 時間はステップ数にほぼ比例する。画質が伴って向上するかは"
          "出力を目視で比較すること。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
