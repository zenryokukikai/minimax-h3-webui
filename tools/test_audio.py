"""音声まわりの動作確認。

  - 既定では動画に音声トラックが入ること
  - audio_format を指定すると音声だけのファイルも取れること
  - audio=false で無音の動画になり、音声ファイルは無いこと

    H3_URL=http://host:18190 python tools/test_audio.py
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_client import H3Client  # noqa: E402

PROMPT = "A candle flame in the dark. Audio: soft crackling and a low hum."
# シードは毎回ランダムにする。固定すると ComfyUI が前回のサンプリング結果を
# キャッシュして末尾しか再実行せず、所要時間の比較が成立しないため。
SIZE = {"width": 352, "height": 192, "steps": 20}
OUT = "/tmp/h3_audio_test"


def streams(path: str) -> str:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,channels,sample_rate",
         "-of", "default=nw=1", path],
        capture_output=True, text=True)
    return " ".join(r.stdout.split())


def fetch(client: H3Client, path: str, dest: str) -> int:
    with urllib.request.urlopen(client.url + path) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    c = H3Client()

    print("== 1) audio_format=mp3 : 動画に音声あり + 音声単体も出力 ==")
    job = c.generate(PROMPT, audio_format="mp3", seed=random.randrange(2**31), **SIZE)
    final = c.wait(job["id"])
    assert final["status"] == "done", final
    assert final.get("audio_url"), "audio_url が返っていない"
    print("   video_url:", final["video_url"])
    print("   audio_url:", final["audio_url"])

    mp4 = c.download(job["id"], f"{OUT}/with_audio.mp4")
    info = streams(mp4)
    print("   mp4:", info)
    assert "audio" in info, "動画に音声トラックが無い"

    n = fetch(c, final["audio_url"], f"{OUT}/audio_only.mp3")
    print(f"   音声単体: {n} bytes | {streams(f'{OUT}/audio_only.mp3')}")
    assert n > 1000, "音声ファイルが小さすぎる"

    print("\n== 2) audio=false : 無音の動画 ==")
    job2 = c.generate(PROMPT, audio=False, seed=random.randrange(2**31), **SIZE)
    final2 = c.wait(job2["id"])
    assert final2["status"] == "done", final2
    mp4b = c.download(job2["id"], f"{OUT}/muted.mp4")
    info2 = streams(mp4b)
    print("   mp4:", info2)
    assert "audio" not in info2, "無音にならず音声トラックが入っている"
    assert not final2.get("audio_url"), "audio_url が付いてしまっている"

    try:
        fetch(c, f"/v1/jobs/{job2['id']}/audio", f"{OUT}/should_fail")
        print("   NG: 音声が取得できてしまった")
        return 1
    except urllib.error.HTTPError as e:
        print(f"   音声取得は期待どおり拒否: HTTP {e.code}")

    print("\n== 3) 所要時間の比較（毎回ランダムシードなのでキャッシュの影響なし）==")
    a, b = final["elapsed_sec"], final2["elapsed_sec"]
    print(f"   音声あり {a}秒 / 無音 {b}秒  差 {a - b:+.1f}秒")
    print("   ※ 音声の生成自体は単一 forward に含まれるため省けない。"
          "無音で省けるのは音声 VAE のデコード分だけ。")

    print("\nすべて成功")
    return 0


if __name__ == "__main__":
    sys.exit(main())
