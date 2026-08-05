"""MiniMax-H3 簡易 Web フロントエンド。

ComfyUI をヘッドレスの生成エンジンとして裏で常駐させ、その HTTP/WS API を
叩く薄いラッパー。利用者にはノードグラフを見せず、プロンプト・解像度・
秒数・参照画像だけの画面を出す。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from aiohttp import web

import workflows

log = logging.getLogger("h3-ui")

STATIC_DIR = Path(__file__).parent / "static"
CLIENT_ID = str(uuid.uuid4())

# prompt_id -> 進捗状態
JOBS: dict[str, dict] = {}


class Comfy:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.session: aiohttp.ClientSession | None = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_json(self, path: str):
        async with self.session.get(f"{self.base}{path}") as r:
            r.raise_for_status()
            return await r.json()

    async def post_json(self, path: str, payload: dict):
        async with self.session.post(f"{self.base}{path}", json=payload) as r:
            body = await r.text()
            if r.status >= 400:
                raise web.HTTPBadGateway(text=body)
            return json.loads(body)

    async def object_info(self):
        return await self.get_json("/api/object_info")

    async def submit(self, prompt: dict) -> str:
        res = await self.post_json("/api/prompt", {"prompt": prompt, "client_id": CLIENT_ID})
        return res["prompt_id"]

    async def history(self, prompt_id: str):
        return await self.get_json(f"/api/history/{prompt_id}")

    async def interrupt(self):
        async with self.session.post(f"{self.base}/api/interrupt") as r:
            return r.status

    async def upload_image(self, filename: str, data: bytes) -> str:
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type="application/octet-stream")
        form.add_field("overwrite", "true")
        async with self.session.post(f"{self.base}/api/upload/image", data=form) as r:
            r.raise_for_status()
            res = await r.json()
        # サブフォルダに入った場合は "sub/name" 形式で返す
        sub = res.get("subfolder") or ""
        return f"{sub}/{res['name']}" if sub else res["name"]


async def ws_listener(comfy: Comfy):
    """ComfyUI の WebSocket を購読して JOBS に進捗を書き込み続ける。"""
    url = comfy.base.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{url}/ws?clientId={CLIENT_ID}"
    while True:
        try:
            async with comfy.session.ws_connect(url, heartbeat=20) as ws:
                log.info("ComfyUI WebSocket 接続")
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    handle_ws_event(payload)
        except Exception as e:  # 落ちても再接続し続ける
            log.warning("WebSocket 切断: %s", e)
        await asyncio.sleep(2)


def handle_ws_event(payload: dict):
    kind = payload.get("type")
    data = payload.get("data") or {}
    pid = data.get("prompt_id")
    job = JOBS.get(pid) if pid else None

    if kind == "execution_start" and job:
        job["state"] = "running"
    elif kind == "progress" and job:
        job["state"] = "running"
        job["step"] = data.get("value", 0)
        job["total"] = data.get("max", 0)
    elif kind == "executing" and job:
        node = data.get("node")
        job["node"] = node
        if node is None:
            job["state"] = "finishing"
    elif kind == "execution_success" and job:
        job["state"] = "done"
    elif kind == "execution_error" and job:
        job["state"] = "error"
        job["error"] = data.get("exception_message") or "実行エラー"
    elif kind == "execution_cached" and job:
        job["state"] = "running"


# ---------------------------------------------------------------- routes


async def index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_config(request):
    comfy: Comfy = request.app["comfy"]
    try:
        oi = await comfy.object_info()
    except Exception as e:
        return web.json_response({"error": f"ComfyUI に接続できません: {e}"}, status=502)

    def combo(node, field):
        """COMBO の選択肢を取り出す。

        ComfyUI には2つの表現が混在する:
          旧: ["<選択肢のリスト>", {...}]          … UNETLoader など
          V3: ["COMBO", {"options": [...]}]       … KSamplerSelect など
        """
        try:
            spec = oi[node]["input"]["required"][field]
        except (KeyError, TypeError):
            return []
        if isinstance(spec[0], list):
            return spec[0]
        if len(spec) > 1 and isinstance(spec[1], dict):
            return spec[1].get("options", [])
        return []

    unets = combo("UNETLoader", "unet_name")
    clips = combo("CLIPLoader", "clip_name")
    vaes = combo("VAELoader", "vae_name")
    d = workflows.DEFAULT_MODELS
    return web.json_response({
        "resolutions": workflows.resolution_options(),
        "cost_model": {
            "c": workflows._COST_C,
            "linear": workflows._COST_LINEAR,
            "quad": workflows._COST_QUAD,
            "baseline_px": workflows._COST_BASELINE_PX,
        },
        "samplers": combo("KSamplerSelect", "sampler_name"),
        "schedulers": combo("BasicScheduler", "scheduler"),
        "available": {"unet": unets, "clip": clips, "vae": vaes},
        "defaults": d,
        "ready": {
            "fl2va": d["unet_fl2va"] in unets,
            "ref2va": d["unet_ref2va"] in unets,
            "clip": d["clip"] in clips,
            "video_vae": d["video_vae"] in vaes,
            "audio_vae": d["audio_vae"] in vaes,
        },
    })


async def api_upload(request):
    comfy: Comfy = request.app["comfy"]
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        raise web.HTTPBadRequest(text="file フィールドがありません")
    data = await field.read()
    name = await comfy.upload_image(field.filename or "upload.png", data)
    return web.json_response({"name": name})


async def api_generate(request):
    comfy: Comfy = request.app["comfy"]
    body = await request.json()

    prompt_text = (body.get("prompt") or "").strip()
    if not prompt_text:
        raise web.HTTPBadRequest(text="プロンプトが空です")

    width = int(body.get("width", 1344))
    height = int(body.get("height", 768))
    seconds = float(body.get("seconds", 5))
    length = workflows.seconds_to_length(seconds)
    steps = int(body.get("steps", 20))
    sampler = body.get("sampler") or "res_multistep"
    scheduler = body.get("scheduler") or "simple"
    seed = body.get("seed")
    seed = random.randint(0, 2**63 - 1) if seed in (None, "", -1) else int(seed)
    mode = body.get("mode", "t2v")

    if mode == "ref2v":
        wf = workflows.build_ref2va(
            prompt_text=prompt_text, width=width, height=height, length=length,
            seed=seed, steps=steps, sampler=sampler, scheduler=scheduler,
            ref_images=body.get("ref_images") or [],
            ref_image_size=body.get("ref_image_size", "match"),
        )
    else:
        wf = workflows.build_fl2va(
            prompt_text=prompt_text, width=width, height=height, length=length,
            seed=seed, steps=steps, sampler=sampler, scheduler=scheduler,
            first_frame=body.get("first_frame") or None,
            last_frame=body.get("last_frame") or None,
        )

    prompt_id = await comfy.submit(wf)
    JOBS[prompt_id] = {
        "state": "queued", "step": 0, "total": steps, "node": None,
        "seed": seed, "length": length,
        "seconds": workflows.length_to_seconds(length),
        "width": width, "height": height,
    }
    return web.json_response({"prompt_id": prompt_id, "seed": seed,
                              "length": length,
                              "seconds": workflows.length_to_seconds(length)})


async def api_status(request):
    comfy: Comfy = request.app["comfy"]
    prompt_id = request.match_info["prompt_id"]
    job = dict(JOBS.get(prompt_id) or {"state": "unknown"})

    if job.get("state") in ("done", "finishing", "unknown"):
        try:
            hist = await comfy.history(prompt_id)
        except Exception:
            hist = {}
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                job["state"] = "error"
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        job["error"] = m[1].get("exception_message", "実行エラー")
            videos = []
            for out in entry.get("outputs", {}).values():
                # SaveVideo は結果を "images" キーに animated=true として返す
                items = (out.get("videos") or []) + (out.get("gifs") or []) + (out.get("images") or [])
                for item in items:
                    if not str(item.get("filename", "")).lower().endswith(
                            (".mp4", ".webm", ".mkv", ".mov", ".gif")):
                        continue
                    videos.append(
                        "/api/view?filename={filename}&subfolder={subfolder}&type={type}".format(
                            filename=item.get("filename", ""),
                            subfolder=item.get("subfolder", ""),
                            type=item.get("type", "output"),
                        )
                    )
            if videos:
                job["state"] = "done"
                job["videos"] = videos
    return web.json_response(job)


async def api_view(request):
    comfy: Comfy = request.app["comfy"]
    qs = request.rel_url.query_string
    async with comfy.session.get(f"{comfy.base}/api/view?{qs}") as r:
        if r.status >= 400:
            raise web.HTTPNotFound()
        resp = web.StreamResponse(
            status=r.status,
            headers={"Content-Type": r.headers.get("Content-Type", "application/octet-stream")},
        )
        await resp.prepare(request)
        async for chunk in r.content.iter_chunked(1 << 16):
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def api_cancel(request):
    comfy: Comfy = request.app["comfy"]
    await comfy.interrupt()
    return web.json_response({"ok": True})


# ---------------------------------------------------------------- app


def make_app(comfy_url: str) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    comfy = Comfy(comfy_url)
    app["comfy"] = comfy

    @asynccontextmanager
    async def lifespan(_):
        await comfy.start()
        task = asyncio.create_task(ws_listener(comfy))
        yield
        task.cancel()
        await comfy.close()

    async def on_startup(a):
        a["_cm"] = lifespan(a)
        await a["_cm"].__aenter__()

    async def on_cleanup(a):
        await a["_cm"].__aexit__(None, None, None)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.add_routes([
        web.get("/", index),
        web.get("/api/config", api_config),
        web.post("/api/upload", api_upload),
        web.post("/api/generate", api_generate),
        web.get("/api/status/{prompt_id}", api_status),
        web.get("/api/view", api_view),
        web.post("/api/cancel", api_cancel),
        web.static("/static", STATIC_DIR),
    ])
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18190)
    p.add_argument("--comfy", default="http://127.0.0.1:18188")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("http://%s:%s  (ComfyUI: %s)", args.host, args.port, args.comfy)
    web.run_app(make_app(args.comfy), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
