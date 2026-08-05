"""MiniMax-H3 Web UI API の Python クライアント。

標準ライブラリのみで動く。ライブラリとしても CLI としても使える。

ライブラリとして:
    from h3_client import H3Client
    c = H3Client("http://your-host:18190", api_key="...")
    job = c.generate("雨の夜の路地。音: 雨音と遠くの雷", width=864, height=480, wait=True)
    c.download(job["id"], "out.mp4")

CLI として:
    python tools/h3_client.py options
    python tools/h3_client.py generate "雨の夜の路地。音: 雨音" -o out.mp4
    python tools/h3_client.py generate "..." --mode i2v --image first.png -o out.mp4
    python tools/h3_client.py status <job-id>

接続先は --url か環境変数 H3_URL、API キーは --api-key か H3_API_KEY。
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


class H3Error(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 0):
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.status = code, message, status


class H3Client:
    def __init__(self, url: str | None = None, api_key: str | None = None,
                 timeout: float = 60.0):
        self.url = (url or os.environ.get("H3_URL", "http://127.0.0.1:18190")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("H3_API_KEY", "")
        self.timeout = timeout

    # ---------------------------------------------------------- 低レベル

    def _request(self, method: str, path: str, payload=None, raw=False, timeout=None):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.read() if raw else json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                err = json.loads(body)["error"]
                raise H3Error(err.get("code", "http_error"), err.get("message", ""), e.code)
            except (ValueError, KeyError):
                raise H3Error("http_error", body.decode("utf-8", "replace")[:500], e.code)

    # ---------------------------------------------------------- 高レベル

    def options(self) -> dict:
        """解像度プリセット・サンプラー・モデル配置状況を取得する。"""
        return self._request("GET", "/v1/options")

    def upload_image(self, path: str) -> str:
        """画像を登録し、生成リクエストで使える参照名を返す。"""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = mimetypes.guess_type(path)[0] or ""
        if not mime.startswith("image/"):
            raise H3Error("invalid_file", f"画像ファイルではありません: {path}")
        return self._request("POST", "/v1/images",
                             {"data": data, "filename": os.path.basename(path)})["ref"]

    def generate(self, prompt: str, *, wait: bool = False, timeout: float = 1800,
                 **params) -> dict:
        """生成を投入する。wait=True なら完了まで待って結果を返す。"""
        body = {"prompt": prompt, **params}
        if wait:
            body.update(wait=True, timeout=timeout)
            # サーバ側の待ちより少し長めに HTTP タイムアウトを取る
            return self._request("POST", "/v1/generate", body, timeout=timeout + 60)
        return self._request("POST", "/v1/generate", body)

    def status(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def jobs(self, limit: int = 20) -> list:
        return self._request("GET", f"/v1/jobs?limit={limit}")["jobs"]

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", f"/v1/jobs/{job_id}/cancel")

    def download(self, job_id: str, dest: str) -> str:
        blob = self._request("GET", f"/v1/jobs/{job_id}/video", raw=True, timeout=600)
        with open(dest, "wb") as f:
            f.write(blob)
        return dest

    def wait(self, job_id: str, *, timeout: float = 1800, poll: float = 2.0,
             on_progress=None) -> dict:
        """ポーリングで完了を待つ（wait=True を使えない場合向け）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = self.status(job_id)
            if on_progress:
                on_progress(s)
            if s["status"] in ("done", "error", "cancelled"):
                return s
            time.sleep(poll)
        raise H3Error("timeout", f"{timeout} 秒以内に完了しませんでした")


# ---------------------------------------------------------------- CLI


def _print_progress(s: dict):
    p = s.get("progress", {})
    el = s.get("elapsed_sec")
    bar = f"{p.get('step', 0)}/{p.get('total', 0)}" if p.get("total") else ""
    sys.stderr.write(f"\r{s['status']:<10} {bar:>8} "
                     f"{('%.0f秒' % el) if el is not None else '':>8}   ")
    sys.stderr.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MiniMax-H3 API クライアント")
    ap.add_argument("--url", default=None, help="既定: $H3_URL か http://127.0.0.1:18190")
    ap.add_argument("--api-key", default=None, help="既定: $H3_API_KEY")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("options", help="解像度プリセットとモデル配置状況を表示")

    g = sub.add_parser("generate", help="動画を生成する")
    g.add_argument("prompt")
    g.add_argument("-o", "--out", help="保存先。省略すると保存せず job id だけ表示")
    g.add_argument("--mode", default="t2v", choices=["t2v", "i2v", "ref2v"])
    g.add_argument("--width", type=int, default=864)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--seconds", type=float, default=5)
    g.add_argument("--steps", type=int, default=20)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--sampler", default=None)
    g.add_argument("--scheduler", default=None)
    g.add_argument("--image", action="append", default=[],
                   help="i2v なら先頭・末尾フレーム、ref2v なら参照画像（複数可）")
    g.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    g.add_argument("--no-wait", action="store_true", help="投入だけして終了する")
    g.add_argument("--timeout", type=float, default=1800)

    s = sub.add_parser("status", help="ジョブの状態を表示")
    s.add_argument("job_id")

    d = sub.add_parser("download", help="生成済み動画を保存")
    d.add_argument("job_id")
    d.add_argument("-o", "--out", required=True)

    c = sub.add_parser("cancel", help="実行中のジョブを中断")
    c.add_argument("job_id")

    j = sub.add_parser("jobs", help="最近のジョブ一覧")
    j.add_argument("--limit", type=int, default=20)

    args = ap.parse_args(argv)
    client = H3Client(args.url, args.api_key)

    try:
        if args.cmd == "options":
            o = client.options()
            print("モデル配置状況:", json.dumps(o["ready"], ensure_ascii=False))
            print("\n解像度プリセット:")
            for r in o["resolutions"]:
                print("  ", r["label"])
            print("\nサンプラー:", ", ".join(o["samplers"][:12]), "…")
            return 0

        if args.cmd == "status":
            print(json.dumps(client.status(args.job_id), ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "jobs":
            for job in client.jobs(args.limit):
                r = job.get("request", {})
                print(f"{job['id']}  {job['status']:<10} "
                      f"{r.get('width')}x{r.get('height')}  seed={r.get('seed')}")
            return 0

        if args.cmd == "cancel":
            print(json.dumps(client.cancel(args.job_id), ensure_ascii=False, indent=2))
            return 0

        if args.cmd == "download":
            print(client.download(args.job_id, args.out))
            return 0

        # generate
        params = {"mode": args.mode, "width": args.width, "height": args.height,
                  "seconds": args.seconds, "steps": args.steps,
                  "ref_image_size": args.ref_image_size}
        if args.seed is not None:
            params["seed"] = args.seed
        if args.sampler:
            params["sampler"] = args.sampler
        if args.scheduler:
            params["scheduler"] = args.scheduler

        refs = [client.upload_image(p) for p in args.image]
        if args.mode == "i2v":
            if not refs:
                print("mode=i2v には --image が必要です", file=sys.stderr)
                return 2
            params["first_frame"] = refs[0]
            if len(refs) > 1:
                params["last_frame"] = refs[1]
        elif args.mode == "ref2v":
            if not refs:
                print("mode=ref2v には --image が必要です", file=sys.stderr)
                return 2
            params["ref_images"] = refs

        job = client.generate(args.prompt, **params)
        print(f"job id: {job['id']}  seed: {job['request']['seed']}  "
              f"{job['request']['seconds']}秒 相対コスト {job.get('relative_cost')}×",
              file=sys.stderr)
        if args.no_wait:
            print(job["id"])
            return 0

        final = client.wait(job["id"], timeout=args.timeout, on_progress=_print_progress)
        sys.stderr.write("\n")
        if final["status"] != "done":
            print(f"失敗: {final.get('error', final['status'])}", file=sys.stderr)
            return 1
        if args.out:
            print(client.download(job["id"], args.out))
        else:
            print(job["id"])
        return 0

    except H3Error as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
