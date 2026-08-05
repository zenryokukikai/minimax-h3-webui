"""キューイングの動作確認。

複数件を続けて投入し、
  - 即座に id が返ること
  - 2件目以降に待ち順位が付くこと
  - キュー待ちの1件だけを取り消しても実行中のものが巻き込まれないこと
  - 完了時に status からダウンロードパスが得られること
を確認する。

    H3_URL=http://host:18190 python tools/test_queue.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_client import H3Client, H3Error  # noqa: E402

client = H3Client()
N = 3
SIZE = (352, 192)   # 最小コストで速く回す


def show_queue(tag: str):
    q = client._request("GET", "/v1/queue")
    run = [j["id"][:8] for j in q["running"]]
    pend = [j["id"][:8] for j in q["pending"]]
    print(f"  [{tag}] 実行中={run} 待機={pend}")
    return q


print(f"== {N} 件を連続投入（{SIZE[0]}x{SIZE[1]}）==")
jobs = []
for i in range(N):
    t0 = time.time()
    j = client.generate(f"Test clip {i + 1}: a spinning glass cube on a dark table. "
                        f"Audio: a soft low hum. No text.",
                        width=SIZE[0], height=SIZE[1], seconds=5, steps=20, seed=1000 + i)
    dt = (time.time() - t0) * 1000
    jobs.append(j["id"])
    print(f"  {i + 1}件目 id={j['id'][:8]} 応答 {dt:.0f}ms "
          f"status={j['status']} queue_position={j.get('queue_position')}")

assert all(jobs), "id が返っていない"
print("\n== キュー状況 ==")
show_queue("投入直後")

print("\n== 待機中の3件目だけ取り消す ==")
res = client.cancel(jobs[2])
print(f"  3件目 status={res['status']}")
time.sleep(2)
q = show_queue("取消後")
running_ids = [j["id"] for j in q["running"]]
assert jobs[2] not in running_ids, "取り消した3件目が実行中にいる"
s0 = client.status(jobs[0])
assert s0["status"] in ("queued", "running", "decoding", "done"), \
    f"1件目が巻き込まれた: {s0['status']}"
print(f"  1件目は無事: status={s0['status']}")

print("\n== 1・2件目の完了を待つ ==")
for n, jid in enumerate(jobs[:2], 1):
    last = None
    while True:
        s = client.status(jid)
        key = (s["status"], s.get("queue_position"), s["progress"]["step"])
        if key != last:
            print(f"  {n}件目 {s['status']:<9} pos={s.get('queue_position')} "
                  f"step={s['progress']['step']}/{s['progress']['total']}")
            last = key
        if s["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(2)
    assert s["status"] == "done", f"{n}件目が失敗: {s.get('error')}"
    assert s.get("video_url"), f"{n}件目に video_url がない"
    print(f"  {n}件目 完了 {s['elapsed_sec']}秒 -> {s['video_url']}")

print("\n== ダウンロード ==")
for n, jid in enumerate(jobs[:2], 1):
    path = client.download(jid, f"/tmp/h3_queue_test_{n}.mp4")
    size = os.path.getsize(path)
    assert size > 10000, f"{path} が小さすぎる ({size} bytes)"
    print(f"  {n}件目 {path} {size / 1024:.0f} KB")

print("\n== 取り消した3件目のダウンロードは失敗すべき ==")
try:
    client.download(jobs[2], "/tmp/h3_queue_test_3.mp4")
    print("  NG: 取得できてしまった")
    sys.exit(1)
except H3Error as e:
    print(f"  OK 期待どおり拒否: {e}")

print("\nすべて成功")
