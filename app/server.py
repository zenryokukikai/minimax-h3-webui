"""MiniMax-H3 の HTTP API と簡易 Web フロントエンド。

ComfyUI をヘッドレスの生成エンジンとして裏で常駐させ、その API を
`/v1/*` の安定したインタフェースとして公開する。同梱の Web UI も
この `/v1` を叩くだけなので、UI とプログラムから見える挙動は同じ。

API の詳細は README、機械可読な定義は GET /v1/openapi.json を参照。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

import workflows

log = logging.getLogger("h3")

STATIC_DIR = Path(__file__).parent / "static"
CLIENT_ID = str(uuid.uuid4())
API_VERSION = "1.0.0"

# 完了済みジョブをこの件数だけ保持する（メモリ上のみ。再起動で消える）
MAX_JOBS = 200

# 入力の許容範囲。ここが唯一の定義箇所で、Web UI は /v1/options 経由で
# この値を読んで入力欄の min/max に反映する（UI 側に数字を持たせない）。
# ステップ数の上限は ComfyUI の BasicScheduler に合わせてある。
# モデル側にステップ数の制約はなく、既定の 20 は公式テンプレートの値。
LIMITS = {
    "steps": {"min": 1, "max": 10000, "default": 20},
    "seconds": {"min": 0.2, "max": 20, "default": 5},
    "dimension": {"min": 32, "max": 4096},
    "ref_images": {"min": 1, "max": 9},
    # MiniMaxH3SigmaShift。映像と音声で別々の flow shift を持つ。
    # 未指定ならノード自体を挟まないのでモデル本来の挙動になる。
    "shift": {"min": 0.01, "max": 100.0,
              "default_video": workflows.DEFAULT_SHIFT_VIDEO,
              "default_audio": workflows.DEFAULT_SHIFT_AUDIO},
    "audio_formats": sorted(workflows.AUDIO_SAVE_NODES),
    # 静止画 (HiDream-O1)。2K ネイティブなので既定を大きめに取る。
    "image_steps": {"min": 1, "max": 200,
                    "default": workflows.IMAGE_DEFAULTS["steps"]},
    "image_size": {"default_width": 1344, "default_height": 768,
                   "reference_megapixels": 1.0},
    # BlockSparseAttention。ComfyUI 側にノードがあるときだけ使える（/v1/options の ready.sparse）。
    "sparse": {"methods": list(workflows.SPARSE_METHODS),
               "tau": {"min": 0.0, "max": 4.0, "default": workflows.SPARSE_DEFAULTS["tau"]},
               "keep_percent": {"min": 0.5, "max": 95.0,
                                "default": workflows.SPARSE_DEFAULTS["keep_percent"]},
               "start_percent": {"min": 0.0, "max": 1.0,
                                 "default": workflows.SPARSE_DEFAULTS["start_percent"]},
               "end_percent": {"min": 0.0, "max": 1.0,
                               "default": workflows.SPARSE_DEFAULTS["end_percent"]}},
}

JOBS: "dict[str, dict]" = {}


# ---------------------------------------------------------------- ComfyUI


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

    async def object_info(self):
        return await self.get_json("/api/object_info")

    async def submit(self, prompt: dict) -> str:
        async with self.session.post(
            f"{self.base}/api/prompt", json={"prompt": prompt, "client_id": CLIENT_ID}
        ) as r:
            body = await r.text()
            if r.status >= 400:
                # ComfyUI のバリデーションエラーをそのまま伝える
                raise ApiError(502, "comfyui_rejected_prompt", body[:2000])
            return json.loads(body)["prompt_id"]

    async def history(self, prompt_id: str):
        return await self.get_json(f"/api/history/{prompt_id}")

    async def interrupt(self):
        """実行中のジョブを中断する（キュー待ちには作用しない）。"""
        async with self.session.post(f"{self.base}/api/interrupt") as r:
            return r.status

    async def queue(self) -> "tuple[list[str], list[str]]":
        """(実行中の prompt_id, キュー待ちの prompt_id) を投入順で返す。

        ComfyUI の /api/queue は [番号, prompt_id, prompt, extra, outputs] の
        配列を返す。番号が投入順なのでそれで整列する。
        """
        q = await self.get_json("/api/queue")

        def ids(key):
            rows = [r for r in q.get(key, []) if isinstance(r, list) and len(r) >= 2]
            return [str(r[1]) for r in sorted(rows, key=lambda r: r[0])]

        return ids("queue_running"), ids("queue_pending")

    async def dequeue(self, prompt_ids: "list[str]"):
        """キュー待ちのジョブを取り消す（実行中には作用しない）。"""
        async with self.session.post(
            f"{self.base}/api/queue", json={"delete": prompt_ids}
        ) as r:
            return r.status

    async def upload_image(self, filename: str, data: bytes) -> str:
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename,
                       content_type="application/octet-stream")
        form.add_field("overwrite", "true")
        async with self.session.post(f"{self.base}/api/upload/image", data=form) as r:
            if r.status >= 400:
                raise ApiError(502, "upload_failed", (await r.text())[:500])
            res = await r.json()
        sub = res.get("subfolder") or ""
        return f"{sub}/{res['name']}" if sub else res["name"]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# ---------------------------------------------------------------- 進捗購読


async def ws_listener(comfy: Comfy):
    """ComfyUI の WebSocket を購読して JOBS に進捗を反映し続ける。"""
    url = comfy.base.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{url}/ws?clientId={CLIENT_ID}"
    while True:
        try:
            async with comfy.session.ws_connect(url, heartbeat=20) as ws:
                log.info("ComfyUI WebSocket 接続")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            handle_ws_event(json.loads(msg.data))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            log.warning("WebSocket 切断: %s", e)
        await asyncio.sleep(2)


def handle_ws_event(payload: dict):
    data = payload.get("data") or {}
    job = JOBS.get(data.get("prompt_id") or "")
    if job is None:
        return
    kind = payload.get("type")

    if kind in ("execution_start", "execution_cached"):
        job["status"] = "running"
        job.setdefault("started_at", time.time())
    elif kind == "progress":
        job["status"] = "running"
        job.setdefault("started_at", time.time())
        job["step"] = data.get("value", 0)
        job["total"] = data.get("max", 0)
    elif kind == "executing" and data.get("node") is None:
        job["status"] = "decoding"
    elif kind == "execution_success":
        job["status"] = "done"
    elif kind == "execution_error":
        job["status"] = "error"
        job["error"] = data.get("exception_message") or "実行エラー"
    if job.get("status") in ("done", "error"):
        job.setdefault("finished_at", time.time())
        job["_event"].set()


def prune_jobs():
    done = [(j.get("finished_at", 0), i) for i, j in JOBS.items()
            if j.get("status") in ("done", "error", "cancelled")]
    if len(done) <= MAX_JOBS:
        return
    for _, jid in sorted(done)[: len(done) - MAX_JOBS]:
        JOBS.pop(jid, None)


# ---------------------------------------------------------------- 表現変換


def job_view(job_id: str, job: dict) -> dict:
    total = job.get("total") or job.get("steps") or 0
    step = job.get("step", 0)
    out = {
        "id": job_id,
        "status": job.get("status", "unknown"),
        "progress": {
            "step": step,
            "total": total,
            "percent": round(step / total * 100, 1) if total else 0.0,
        },
        "request": {
            "prompt": job.get("prompt"),
            "mode": job.get("mode"),
            "width": job.get("width"),
            "height": job.get("height"),
            "seconds": job.get("seconds"),
            "length": job.get("length"),
            "steps": job.get("steps"),
            "seed": job.get("seed"),
            "sampler": job.get("sampler"),
            "scheduler": job.get("scheduler"),
            "audio": job.get("audio"),
            "audio_format": job.get("audio_format"),
            "shift_video": job.get("shift_video"),
            "shift_audio": job.get("shift_audio"),
            "prep_image": job.get("prep_image"),
            "sparse": job.get("sparse"),
        },
        "relative_cost": job.get("relative_cost"),
        "created_at": job.get("created_at"),
    }
    if job.get("queue_position") is not None:
        # 0 = 実行中、1 以上 = 自分より前に残っている件数
        out["queue_position"] = job["queue_position"]
    if job.get("started_at"):
        end = job.get("finished_at") or time.time()
        out["elapsed_sec"] = round(end - job["started_at"], 1)
    if job.get("queued_at") and not job.get("started_at"):
        out["queued_sec"] = round(time.time() - job["queued_at"], 1)
    if job.get("video_url"):
        out["video_url"] = job["video_url"]
    if job.get("audio_url"):
        out["audio_url"] = job["audio_url"]
    if job.get("image_url"):
        # prep_image を使った動画では、参照に使った静止画がここに入る
        out["image_url"] = job["image_url"]
    if job.get("error"):
        out["error"] = job["error"]
    return out


async def refresh_queue_position(comfy: Comfy, job_ids: "list[str]"):
    """未完了ジョブに ComfyUI のキュー内順位を書き込む。

    ComfyUI は一度に1件しか実行しないので、投入は自動的に直列化される。
    ここではその待ち行列を利用者に見せるためだけに位置を引く。
    """
    targets = [i for i in job_ids
               if JOBS.get(i, {}).get("status") in ("queued", "running")]
    if not targets:
        return
    try:
        running, pending = await comfy.queue()
    except Exception:
        return
    for jid in targets:
        job = JOBS[jid]
        if jid in running:
            job["queue_position"] = 0
        elif jid in pending:
            job["queue_position"] = pending.index(jid) + 1
            job["status"] = "queued"
        else:
            # キューにいない = 実行済みか取り消し済み。履歴側で判定する
            job["queue_position"] = None


async def refresh_from_history(comfy: Comfy, job_id: str, job: dict):
    """ComfyUI の履歴から成果物 URL と最終状態を拾う。"""
    if job.get("video_url") or job.get("status") == "error":
        return
    try:
        entry = (await comfy.history(job_id)).get(job_id)
    except Exception:
        return
    if not entry:
        return

    status = entry.get("status", {})
    if status.get("status_str") == "error":
        job["status"] = "error"
        for m in status.get("messages", []):
            if m[0] == "execution_error":
                job["error"] = m[1].get("exception_message", "実行エラー")
        job.setdefault("finished_at", time.time())
        job["_event"].set()
        return

    for out in entry.get("outputs", {}).values():
        # SaveVideo は成果物を "images" キーに animated=true として返す
        items = ((out.get("videos") or []) + (out.get("gifs") or [])
                 + (out.get("images") or []) + (out.get("audio") or []))
        for item in items:
            name = str(item.get("filename", ""))
            ref = {"filename": name,
                   "subfolder": item.get("subfolder", ""),
                   "type": item.get("type", "output")}
            low = name.lower()
            if low.endswith((".mp4", ".webm", ".mkv", ".mov", ".gif")):
                job["video_url"] = f"/v1/jobs/{job_id}/video"
                job["_comfy_file"] = ref
            elif low.endswith((".flac", ".mp3", ".opus", ".wav")):
                job["audio_url"] = f"/v1/jobs/{job_id}/audio"
                job["_comfy_audio"] = ref
            elif low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                job["image_url"] = f"/v1/jobs/{job_id}/image"
                job["_comfy_image"] = ref

    if job.get("video_url") or (job.get("mode") == "t2i" and job.get("image_url")):
        job["status"] = "done"
        job.setdefault("finished_at", time.time())
        job["_event"].set()
        return


# ---------------------------------------------------------------- 入力検証


def _int(body: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = body.get(key, default)
    if v is None:
        v = default
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_parameter", f"{key} は整数で指定してください")
    if not lo <= v <= hi:
        raise ApiError(400, "invalid_parameter", f"{key} は {lo}〜{hi} の範囲で指定してください")
    return v


def _seed(body: dict, key: str = "seed") -> int:
    v = body.get(key)
    if v in (None, "", -1):
        return random.randint(0, 2**63 - 1)
    return _int({key: v}, key, 0, 0, 2**63 - 1)


def _sparse(spec) -> dict | None:
    """sparse: true / "sol-attn" / {"method", "tau"|"keep_percent", "start_percent", "end_percent"}。"""
    if spec in (None, False, ""):
        return None
    if spec is True:
        spec = {}
    elif isinstance(spec, str):
        spec = {"method": spec}
    if not isinstance(spec, dict):
        raise ApiError(400, "invalid_parameter",
                       "sparse は true / 手法名 / 設定オブジェクトで指定してください")
    lim = LIMITS["sparse"]
    method = spec.get("method") or "sol-attn"
    if method not in lim["methods"]:
        raise ApiError(400, "invalid_parameter",
                       f"sparse.method は {lim['methods']} のいずれかです")
    out = {"method": method, "min_tokens": workflows.SPARSE_DEFAULTS["min_tokens"]}
    keys = ("start_percent", "end_percent",
            "tau" if method == "sol-attn" else "keep_percent")
    for key in keys:
        v = spec.get(key, lim[key]["default"])
        if isinstance(v, bool):
            raise ApiError(400, "invalid_parameter", f"sparse.{key} は数値で指定してください")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ApiError(400, "invalid_parameter", f"sparse.{key} は数値で指定してください")
        if not lim[key]["min"] <= v <= lim[key]["max"]:
            raise ApiError(400, "invalid_parameter",
                           f"sparse.{key} は {lim[key]['min']}〜{lim[key]['max']} の範囲です")
        out[key] = v
    if out["start_percent"] >= out["end_percent"]:
        raise ApiError(400, "invalid_parameter",
                       "sparse.start_percent は end_percent より小さくしてください")
    return out


def parse_request(body: dict) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_body", "JSON オブジェクトを送ってください")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ApiError(400, "missing_prompt", "prompt は必須です")

    mode = body.get("mode") or "t2v"
    if mode not in ("t2v", "i2v", "ref2v", "t2i"):
        raise ApiError(400, "invalid_mode",
                       "mode は t2v / i2v / ref2v / t2i のいずれかです")

    dim = LIMITS["dimension"]
    width = workflows.snap_dimension(_int(body, "width", 864, dim["min"], dim["max"]))
    height = workflows.snap_dimension(_int(body, "height", 480, dim["min"], dim["max"]))

    sec_lim = LIMITS["seconds"]
    if mode == "t2i":
        img_lim = LIMITS["image_steps"]
        return {
            "mode": "t2i",
            "prompt": prompt,
            "width": width if body.get("width") else LIMITS["image_size"]["default_width"],
            "height": height if body.get("height") else LIMITS["image_size"]["default_height"],
            "steps": _int(body, "steps", img_lim["default"], img_lim["min"], img_lim["max"]),
            "seed": _seed(body),
            "negative": body.get("negative") or "",
        }

    seconds = body.get("seconds", sec_lim["default"])
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_parameter", "seconds は数値で指定してください")
    if not sec_lim["min"] <= seconds <= sec_lim["max"]:
        raise ApiError(400, "invalid_parameter",
                       f"seconds は {sec_lim['min']}〜{sec_lim['max']} の範囲で指定してください")

    req = {
        "prompt": prompt,
        "mode": mode,
        "width": width,
        "height": height,
        "seconds": seconds,
        "length": workflows.seconds_to_length(seconds),
        "steps": _int(body, "steps", LIMITS["steps"]["default"],
                      LIMITS["steps"]["min"], LIMITS["steps"]["max"]),
        "seed": _seed(body),
        "sampler": body.get("sampler") or "res_multistep",
        "scheduler": body.get("scheduler") or "simple",
    }

    if mode == "i2v":
        if not body.get("first_frame"):
            raise ApiError(400, "missing_image", "mode=i2v には first_frame が必要です")
        req["first_frame"] = body["first_frame"]
        req["last_frame"] = body.get("last_frame") or None
    elif mode == "ref2v":
        refs = body.get("ref_images") or []
        if not isinstance(refs, list) or not refs:
            raise ApiError(400, "missing_image", "mode=ref2v には ref_images が必要です")
        if len(refs) > LIMITS["ref_images"]["max"]:
            raise ApiError(400, "too_many_images",
                           f"ref_images は最大{LIMITS['ref_images']['max']}枚です")
        req["ref_images"] = refs
        size = body.get("ref_image_size", "match")
        if size not in ("match", "max"):
            raise ApiError(400, "invalid_parameter", "ref_image_size は match / max です")
        req["ref_image_size"] = size

    # 音声。H3 は映像と音声を単一 forward で同時生成するので「音声を作らない」
    # 選択肢はない。audio=false は出力動画に音声トラックを入れないという意味。
    req["audio"] = bool(body.get("audio", True))

    fmt = body.get("audio_format")
    if fmt not in (None, "", *workflows.AUDIO_SAVE_NODES):
        raise ApiError(400, "invalid_parameter",
                       f"audio_format は {sorted(workflows.AUDIO_SAVE_NODES)} "
                       f"のいずれかです")
    req["audio_format"] = fmt or None

    lim = LIMITS["shift"]
    for key in ("shift_video", "shift_audio"):
        v = body.get(key)
        if v is None:
            req[key] = None
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ApiError(400, "invalid_parameter", f"{key} は数値で指定してください")
        if not lim["min"] <= v <= lim["max"]:
            raise ApiError(400, "invalid_parameter",
                           f"{key} は {lim['min']}〜{lim['max']} の範囲です")
        req[key] = v

    # 参照用の静止画を先に作ってから動画にするオプション。
    # 同じ ComfyUI グラフに両方を入れて IMAGE を直結するので、
    # 中間ファイルの受け渡しは発生せず、キューにも1件しか積まれない。
    prep = body.get("prep_image")
    if prep:
        if prep is True:
            prep = {}
        if not isinstance(prep, dict):
            raise ApiError(400, "invalid_parameter",
                           "prep_image は true か設定オブジェクトで指定してください")
        if mode == "i2v" and body.get("first_frame"):
            raise ApiError(400, "invalid_parameter",
                           "first_frame を指定した場合 prep_image は使えません")
        img_lim = LIMITS["image_steps"]
        iw, ih = workflows.image_size_for(
            width, height, LIMITS["image_size"]["reference_megapixels"])
        req["prep_image"] = {
            # 既定では動画と同じプロンプトを使う（映像の記述がそのまま効く）
            "prompt": (prep.get("prompt") or "").strip() or prompt,
            "negative": prep.get("negative") or "",
            "width": workflows.snap_dimension(int(prep.get("width") or iw)),
            "height": workflows.snap_dimension(int(prep.get("height") or ih)),
            "steps": _int(prep, "steps", img_lim["default"],
                          img_lim["min"], img_lim["max"]),
            "seed": _seed(prep),
            "save": bool(prep.get("save", True)),
        }
        if mode == "i2v":
            # 生成画像を先頭フレームにするので i2v と同じ扱いになる
            req["mode"] = mode = "i2v"
    else:
        req["prep_image"] = None

    req["sparse"] = _sparse(body.get("sparse"))

    req["seconds"] = workflows.length_to_seconds(req["length"])
    req["relative_cost"] = round(workflows.relative_cost(width, height), 3)
    return req


def build_workflow(req: dict) -> dict:
    if req["mode"] == "t2i":
        return workflows.build_t2i(
            prompt_text=req["prompt"], width=req["width"], height=req["height"],
            seed=req["seed"], steps=req["steps"], negative=req["negative"])

    common = dict(
        prompt_text=req["prompt"], width=req["width"], height=req["height"],
        length=req["length"], seed=req["seed"], steps=req["steps"],
        sampler=req["sampler"], scheduler=req["scheduler"],
        audio=req["audio"], audio_format=req["audio_format"],
        shift_video=req["shift_video"], shift_audio=req["shift_audio"],
        prep_image=req["prep_image"],
        sparse=req.get("sparse"),
    )
    if req["mode"] == "ref2v":
        return workflows.build_ref2va(
            ref_images=req["ref_images"], ref_image_size=req["ref_image_size"], **common)
    return workflows.build_fl2va(
        first_frame=req.get("first_frame"), last_frame=req.get("last_frame"), **common)


# ---------------------------------------------------------------- v1 API


@web.middleware
async def middleware(request, handler):
    """API キー検証と、ApiError の JSON 化。"""
    key = request.app["api_key"]
    path = request.path
    if key and path.startswith("/v1/") and path != "/v1/openapi.json":
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else request.headers.get("X-API-Key", "")
        if token != key:
            return web.json_response(
                {"error": {"code": "unauthorized", "message": "API キーが必要です"}}, status=401)
    try:
        return await handler(request)
    except ApiError as e:
        return web.json_response(
            {"error": {"code": e.code, "message": e.message}}, status=e.status)
    except web.HTTPException:
        raise
    except Exception as e:
        log.exception("未処理の例外")
        return web.json_response(
            {"error": {"code": "internal_error", "message": str(e)}}, status=500)


async def v1_options(request):
    """生成に使える選択肢と、モデルの配置状況を返す。"""
    comfy: Comfy = request.app["comfy"]
    try:
        oi = await comfy.object_info()
    except Exception as e:
        raise ApiError(502, "comfyui_unreachable", f"ComfyUI に接続できません: {e}")

    def combo(node, field):
        """ComfyUI には2つの COMBO 表現が混在する。

        旧: ["<選択肢のリスト>", {...}]        … UNETLoader など
        V3: ["COMBO", {"options": [...]}]      … KSamplerSelect など
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

    unets, clips, vaes = (combo("UNETLoader", "unet_name"),
                          combo("CLIPLoader", "clip_name"),
                          combo("VAELoader", "vae_name"))
    ckpts = combo("CheckpointLoaderSimple", "ckpt_name")
    d = workflows.DEFAULT_MODELS
    return web.json_response({
        "api_version": API_VERSION,
        "resolutions": workflows.resolution_options(),
        "image_resolutions": workflows.image_resolution_options(),
        "samplers": combo("KSamplerSelect", "sampler_name"),
        "schedulers": combo("BasicScheduler", "scheduler"),
        "cost_model": {
            "c": workflows._COST_C,
            "linear": workflows._COST_LINEAR,
            "quad": workflows._COST_QUAD,
            "baseline_px": workflows._COST_BASELINE_PX,
        },
        "models": d,
        "ready": {
            "t2v": d["unet_fl2va"] in unets,
            "i2v": d["unet_fl2va"] in unets,
            "ref2v": d["unet_ref2va"] in unets,
            "text_encoder": d["clip"] in clips,
            "video_vae": d["video_vae"] in vaes,
            "audio_vae": d["audio_vae"] in vaes,
            # 静止画生成。prep_image（参照画像の自動生成）にも必要。
            "t2i": d["image_checkpoint"] in ckpts,
            # BlockSparseAttention（ComfyUI 0.37 以降）。無い環境で sparse を指定すると ComfyUI が拒否する。
            "sparse": "BlockSparseAttention" in oi,
        },
        "limits": {**LIMITS, "fps": workflows.FPS,
                   "canvas_multiple": workflows.CANVAS_MULTIPLE},
    })


async def v1_images(request):
    """画像を登録し、生成リクエストで使える参照名を返す。

    multipart（フィールド名 file）と JSON {"data": "<base64>"} の両対応。
    """
    comfy: Comfy = request.app["comfy"]
    ctype = request.headers.get("Content-Type", "")

    if ctype.startswith("multipart/"):
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            raise ApiError(400, "missing_file", "file フィールドが必要です")
        filename, data = field.filename or "upload.png", await field.read()
    else:
        body = await request.json()
        raw = body.get("data")
        if not raw:
            raise ApiError(400, "missing_file", "data に base64 の画像を入れてください")
        if "," in raw[:64] and raw.lstrip().startswith("data:"):
            raw = raw.split(",", 1)[1]  # data URL を許容
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise ApiError(400, "invalid_base64", "data が base64 として不正です")
        filename = body.get("filename") or "upload.png"

    if not data:
        raise ApiError(400, "empty_file", "画像が空です")
    return web.json_response({"ref": await comfy.upload_image(filename, data)})


async def v1_generate(request):
    """生成を投入する。wait=true なら完了までブロックして結果を返す。"""
    comfy: Comfy = request.app["comfy"]
    body = await request.json()
    req = parse_request(body)

    # ComfyUI は一度に1件しか実行しないので、投入した時点で自動的にキューに並ぶ。
    now = time.time()
    job_id = await comfy.submit(build_workflow(req))
    JOBS[job_id] = {**req, "status": "queued", "step": 0, "total": req["steps"],
                    "created_at": now, "queued_at": now, "_event": asyncio.Event()}
    prune_jobs()
    job = JOBS[job_id]

    if not body.get("wait"):
        # 既定は非同期。待たずに id を返す
        await refresh_queue_position(comfy, [job_id])
        return web.json_response(job_view(job_id, job), status=202)

    timeout = float(body.get("timeout") or 1800)
    try:
        await asyncio.wait_for(job["_event"].wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ApiError(504, "timeout",
                       f"{timeout:.0f} 秒以内に完了しませんでした。"
                       f"GET /v1/jobs/{job_id} で継続確認できます")
    await refresh_from_history(comfy, job_id, job)
    view = job_view(job_id, job)
    if job.get("status") == "error":
        return web.json_response(view, status=500)
    return web.json_response(view)


async def v1_job(request):
    comfy: Comfy = request.app["comfy"]
    job_id = request.match_info["job_id"]
    job = JOBS.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", "そのジョブは存在しないか、保持期間を過ぎています")
    if job.get("status") in ("queued", "running"):
        await refresh_queue_position(comfy, [job_id])
    if job.get("status") in ("done", "decoding", "unknown"):
        await refresh_from_history(comfy, job_id, job)
    return web.json_response(job_view(job_id, job))


async def v1_jobs(request):
    comfy: Comfy = request.app["comfy"]
    items = sorted(JOBS.items(), key=lambda kv: kv[1].get("created_at", 0), reverse=True)
    limit = min(int(request.query.get("limit", 20)), 100)
    items = items[:limit]
    await refresh_queue_position(comfy, [i for i, _ in items])
    return web.json_response({"jobs": [job_view(i, j) for i, j in items]})


async def v1_queue(request):
    """キューの現況。実行中1件＋待機中を投入順で返す。"""
    comfy: Comfy = request.app["comfy"]
    try:
        running, pending = await comfy.queue()
    except Exception as e:
        raise ApiError(502, "comfyui_unreachable", str(e))

    def brief(jid, position):
        job = JOBS.get(jid)
        if job is None:
            # この UI 以外（ComfyUI のノードエディタなど）から投入されたもの
            return {"id": jid, "queue_position": position, "external": True}
        return {**job_view(jid, job), "queue_position": position}

    return web.json_response({
        "running": [brief(i, 0) for i in running],
        "pending": [brief(i, n + 1) for n, i in enumerate(pending)],
        "pending_count": len(pending),
    })


async def _stream_artifact(request, slot: str, fallback_type: str):
    comfy: Comfy = request.app["comfy"]
    job_id = request.match_info["job_id"]
    job = JOBS.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", "そのジョブは存在しません")
    await refresh_from_history(comfy, job_id, job)
    ref = job.get(slot)
    if not ref:
        if slot in ("_comfy_audio", "_comfy_image") and job.get("status") == "done":
            raise ApiError(404, "no_such_artifact",
                           "そのファイルは出力されていません"
                           "（音声は audio_format、静止画は t2i / prep_image が必要）")
        raise ApiError(409, "not_ready",
                       f"まだ成果物がありません (status={job.get('status')})")

    # ?download=1 でブラウザに保存させる（curl -OJ でも使える）
    disposition = "attachment" if request.query.get("download") else "inline"
    qs = "&".join(f"{k}={aiohttp.helpers.quote(str(v), safe='')}" for k, v in ref.items())
    async with comfy.session.get(f"{comfy.base}/api/view?{qs}") as r:
        if r.status >= 400:
            raise ApiError(502, "fetch_failed", "ComfyUI から成果物を取得できません")
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": r.headers.get("Content-Type", fallback_type),
            "Content-Disposition": f'{disposition}; filename="{ref["filename"]}"',
        })
        await resp.prepare(request)
        async for chunk in r.content.iter_chunked(1 << 16):
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def v1_video(request):
    return await _stream_artifact(request, "_comfy_file", "video/mp4")


async def v1_image(request):
    """静止画を取り出す（mode=t2i、または prep_image で作った参照画像）。"""
    return await _stream_artifact(request, "_comfy_image", "image/png")


async def v1_audio(request):
    """音声だけを取り出す（生成時に audio_format を指定した場合）。"""
    return await _stream_artifact(request, "_comfy_audio", "audio/flac")


async def v1_cancel(request):
    comfy: Comfy = request.app["comfy"]
    job_id = request.match_info["job_id"]
    job = JOBS.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", "そのジョブは存在しません")
    if job.get("status") in ("done", "error", "cancelled"):
        return web.json_response(job_view(job_id, job))

    # interrupt は「今動いているもの」を止めるだけなので、キュー待ちの
    # ジョブに対して呼ぶと無関係な実行中ジョブを巻き込む。
    # 対象がどこにいるかを見てから使い分ける。
    try:
        running, pending = await comfy.queue()
    except Exception as e:
        raise ApiError(502, "comfyui_unreachable", str(e))

    if job_id in pending:
        await comfy.dequeue([job_id])
    elif job_id in running:
        await comfy.interrupt()
    # どちらにもいなければ既に終わっている。履歴を見て確定させる
    else:
        await refresh_from_history(comfy, job_id, job)
        if job.get("status") == "done":
            return web.json_response(job_view(job_id, job))

    job["status"] = "cancelled"
    job["queue_position"] = None
    job.setdefault("finished_at", time.time())
    job["_event"].set()
    return web.json_response(job_view(job_id, job))


async def v1_openapi(request):
    return web.json_response(OPENAPI)


# ---------------------------------------------------------------- 画面


async def index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def healthz(request):
    comfy: Comfy = request.app["comfy"]
    try:
        await comfy.get_json("/api/system_stats")
        return web.json_response({"status": "ok", "api_version": API_VERSION})
    except Exception as e:
        return web.json_response({"status": "degraded", "detail": str(e)}, status=503)


OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "MiniMax-H3 Web UI API", "version": API_VERSION,
             "description": "動画とステレオ音声を同時生成する MiniMax-H3 の HTTP API。"},
    "paths": {
        "/v1/options": {"get": {"summary": "解像度プリセット・サンプラー・モデル配置状況"}},
        "/v1/images": {"post": {"summary": "参照画像を登録し ref を得る（multipart または base64）"}},
        "/v1/generate": {"post": {"summary": "生成を投入。wait=true で完了までブロック"}},
        "/v1/jobs": {"get": {"summary": "最近のジョブ一覧"}},
        "/v1/queue": {"get": {"summary": "キューの現況（実行中1件＋待機中を投入順で）"}},
        "/v1/jobs/{job_id}": {"get": {"summary": "ジョブの状態と進捗"}},
        "/v1/jobs/{job_id}/video": {"get": {"summary": "生成された mp4"}},
        "/v1/jobs/{job_id}/audio": {"get": {"summary": "音声のみ（audio_format 指定時）"}},
        "/v1/jobs/{job_id}/image": {"get": {"summary": "静止画（mode=t2i / prep_image 使用時）"}},
        "/v1/jobs/{job_id}/cancel": {"post": {"summary": "実行中のジョブを中断"}},
        "/healthz": {"get": {"summary": "ヘルスチェック"}},
    },
}


def make_app(comfy_url: str, api_key: str = "") -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024, middlewares=[middleware])
    comfy = Comfy(comfy_url)
    app["comfy"] = comfy
    app["api_key"] = api_key

    async def on_startup(a):
        await comfy.start()
        a["_ws"] = asyncio.create_task(ws_listener(comfy))

    async def on_cleanup(a):
        a["_ws"].cancel()
        await comfy.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.add_routes([
        web.get("/", index),
        web.get("/healthz", healthz),
        web.get("/v1/openapi.json", v1_openapi),
        web.get("/v1/options", v1_options),
        web.post("/v1/images", v1_images),
        web.post("/v1/generate", v1_generate),
        web.get("/v1/jobs", v1_jobs),
        web.get("/v1/queue", v1_queue),
        web.get("/v1/jobs/{job_id}", v1_job),
        web.get("/v1/jobs/{job_id}/video", v1_video),
        web.get("/v1/jobs/{job_id}/audio", v1_audio),
        web.get("/v1/jobs/{job_id}/image", v1_image),
        web.post("/v1/jobs/{job_id}/cancel", v1_cancel),
        web.static("/static", STATIC_DIR),
    ])
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18190)
    p.add_argument("--comfy", default="http://127.0.0.1:18188")
    p.add_argument("--api-key", default=os.environ.get("H3_API_KEY", ""),
                   help="指定すると /v1/* に Authorization: Bearer が必要になる")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("http://%s:%s  (ComfyUI: %s, API キー: %s)",
             args.host, args.port, args.comfy, "あり" if args.api_key else "なし")
    web.run_app(make_app(args.comfy, args.api_key), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
