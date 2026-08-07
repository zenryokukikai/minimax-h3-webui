"""MiniMax-H3 の ComfyUI API 形式ワークフローを組み立てる。

ComfyUI 0.30.0 の公式テンプレート (video_minimax_h3_{t2v,i2v,r2v}.json) の
サブグラフを API 形式に展開したもの。ノード入力名は /api/object_info と
comfy_extras/nodes_minimax_h3.py で確認済み。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FPS = 24
CANVAS_MULTIPLE = 32

# MiniMaxH3SigmaShift ノードの既定値。映像と音声で別々の flow shift を持つ。
# 指定がなければこのノード自体を挟まないので、モデル本来の挙動になる。
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

# 既定は Ampere (A100 等) 向けの INT8 ConvRot。
# NVFP4 は Blackwell 専用命令、FP8 は Ada 以降なので、Ampere で
# ネイティブに動く量子化形式は INT8 ConvRot だけになる。
#
# Blackwell (RTX 50xx / GB10 / B200) では NVFP4 がネイティブなので
# models.json で差し替える。速度差は数%だがファイルサイズが約半分になる。
_BUILTIN_MODELS = {
    "unet_fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "unet_ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    # 静止画生成 (HiDream-O1)。動画の参照画像を作るのにも使う。
    # A100 のような Ampere では FP8/MXFP8 がエミュレーションになるので BF16 を選ぶ。
    "image_checkpoint": "hidream_o1_image_dev_bf16.safetensors",
}

# HiDream-O1 Dev の既定サンプリング設定（公式テンプレート image_hidream_o1_dev.json）
IMAGE_DEFAULTS = {
    "steps": 28,
    "cfg": 1.0,
    "scheduler": "normal",
    "noise_scale": 7.6,      # ModelNoiseScale
    "lcm_s_noise": 1.0,
    "lcm_s_noise_end": 1.0,
    "lcm_noise_clip_std": 2.5,
}


def _load_models() -> dict:
    """`models.json` があればファイル名を上書きする（ホストごとの差分吸収）。"""
    path = Path(os.environ.get("H3_MODELS_JSON", Path(__file__).parent / "models.json"))
    if not path.is_file():
        return dict(_BUILTIN_MODELS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # `_` 始まりのキーはコメント用として無視する
    override = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = set(override) - set(_BUILTIN_MODELS)
    if unknown:
        raise ValueError(
            f"{path} に未知のキー: {sorted(unknown)} / "
            f"有効なキー: {sorted(_BUILTIN_MODELS)}")
    return {**_BUILTIN_MODELS, **override}


DEFAULT_MODELS = _load_models()

# 短辺 768px がネイティブ。幅・高さとも 32 の倍数であること。
#
# `cost` は 864x480 を 1.00 とした**相対的な**生成コストの目安。
# 絶対的な所要時間は GPU によって桁で変わるのでここには持たせない
# （同じ設定でも A100 80GB と DGX Spark で3倍以上違う）。
# 同一画素数のプリセットには同じ値を与え、間は画素数から補間している。
#
# `quality` は同一プロンプト・同一シードで中間フレームを目視比較した結果:
#   ok    … 構図・被写体とも破綻なし
#   ghost … 半透明のゴースト・二重像が乗る
#   rough … 被写体の形状が崩れる。構図の当たりを取る下書き用途のみ
RESOLUTION_PRESETS = [
    {"width": 1344, "height": 768, "quality": "ok", "note": "ネイティブ・最高品質"},
    {"width": 1152, "height": 640, "quality": "ok", "note": ""},
    {"width": 864, "height": 480, "quality": "ok", "note": "バランス型（コスト基準）"},
    {"width": 768, "height": 448, "quality": "ok", "note": ""},
    {"width": 672, "height": 384, "quality": "ok", "note": "短辺384px・公式の下限"},
    {"width": 576, "height": 320, "quality": "ok", "note": "実用下限"},
    {"width": 448, "height": 256, "quality": "ghost", "note": "試写用"},
    {"width": 384, "height": 256, "quality": "ghost", "note": "試写用"},
    {"width": 352, "height": 192, "quality": "rough", "note": "下書き用"},
    {"width": 768, "height": 1344, "quality": "ok", "note": "縦"},
    {"width": 640, "height": 1152, "quality": "ok", "note": "縦"},
    {"width": 320, "height": 576, "quality": "ok", "note": "縦・実用下限"},
    {"width": 992, "height": 992, "quality": "ok", "note": "正方形"},
    {"width": 448, "height": 448, "quality": "ok", "note": "正方形・軽量"},
    {"width": 1152, "height": 864, "quality": "ok", "note": ""},
    {"width": 576, "height": 448, "quality": "ok", "note": "軽量"},
    {"width": 864, "height": 1152, "quality": "ok", "note": "縦"},
    {"width": 1344, "height": 576, "quality": "ok", "note": "シネスコ"},
]

_QUALITY_MARK = {"ok": "", "ghost": "⚠ゴーストあり", "rough": "⚠形状が崩れる"}

# 生成コストは実測上ほぼ画素数だけで決まる。
#   固定オーバーヘッド + MLP の線形項 + アテンションの二次項
# A100 80GB / 20ステップ / 124フレーム で測った 1ステップあたり秒数
#   0.068MP:0.91  0.115:1.40  0.184:2.10  0.258:3.00
#   0.344:4.00    0.737:10.0  1.032:16.9
# に上式を当てはめた係数（残差はおおむね5%以内）。
# 係数の絶対値は GPU 依存なので、UI では 864x480 を 1.00 とした比だけを出す。
_COST_C, _COST_LINEAR, _COST_QUAD = 0.332, 7.968, 7.836
_COST_BASELINE_PX = 864 * 480


def _raw_cost(pixels: int) -> float:
    mp = pixels / 1_000_000
    return _COST_C + _COST_LINEAR * mp + _COST_QUAD * mp * mp


def relative_cost(width: int, height: int) -> float:
    """864x480 を 1.00 とした相対生成コスト。"""
    return _raw_cost(width * height) / _raw_cost(_COST_BASELINE_PX)


_COMMON_ASPECTS = {"21:9": (21, 9), "16:9": (16, 9), "3:2": (3, 2), "4:3": (4, 3),
                   "1:1": (1, 1), "3:4": (3, 4), "2:3": (2, 3), "9:16": (9, 16)}


def _aspect(w: int, h: int) -> str:
    """よくある比に近ければその名前で表す。

    キャンバスは32の倍数に丸められるため 1344x768 のように厳密には
    16:9 (1.778) ではなく 1.75 になる。4%以内なら「≈16:9」と表記する。
    """
    from math import gcd
    ratio = w / h
    for label, (x, y) in _COMMON_ASPECTS.items():
        target = x / y
        if abs(ratio - target) / target < 1e-9:
            return label
        if abs(ratio - target) / target < 0.04:
            return "≈" + label
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def resolution_options() -> list[dict]:
    """UI 用にラベルを組み立てたプリセット一覧を返す。"""
    out = []
    for p in RESOLUTION_PRESETS:
        cost = relative_cost(p["width"], p["height"])
        parts = [f"{_aspect(p['width'], p['height']):>5}",
                 f"{p['width']}x{p['height']}".ljust(9),
                 f"コスト {cost:.2f}×"]
        tail = " ".join(x for x in (_QUALITY_MARK[p["quality"]], p["note"]) if x)
        if tail:
            parts.append(tail)
        out.append({**p, "cost": round(cost, 3), "label": "  ".join(parts)})
    return out


# HiDream-O1 は 2K ネイティブ。動画側と違い時間軸が無いので画素数だけが効く。
IMAGE_RESOLUTION_PRESETS = [
    {"width": 2048, "height": 2048, "note": "ネイティブ 2K・正方形"},
    {"width": 2048, "height": 1152, "note": "ネイティブ 2K・横"},
    {"width": 1152, "height": 2048, "note": "ネイティブ 2K・縦"},
    {"width": 1344, "height": 768, "note": "動画のネイティブ解像度に一致"},
    {"width": 768, "height": 1344, "note": "動画のネイティブ解像度に一致・縦"},
    {"width": 1024, "height": 1024, "note": "軽量・正方形"},
    {"width": 1344, "height": 896, "note": ""},
    {"width": 896, "height": 1344, "note": "縦"},
]


def image_resolution_options() -> list[dict]:
    out = []
    for p in IMAGE_RESOLUTION_PRESETS:
        label = f"{_aspect(p['width'], p['height']):>5}  {p['width']}x{p['height']}".ljust(22)
        if p["note"]:
            label += "  " + p["note"]
        out.append({**p, "label": label})
    return out


def seconds_to_length(seconds: float) -> int:
    """秒数を H3 が受け付けるフレーム長 (24fps, 17k+5 グリッド) に切り上げる。

    公式テンプレートの ComfyMathExpression と同じ式:
        max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17
    """
    length = max(5, round(seconds * FPS))
    return length + (5 - (length % 17)) % 17


def length_to_seconds(length: int) -> float:
    return round(length / FPS, 2)


def snap_dimension(value: int) -> int:
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _loaders(unet_name: str, clip_name: str, video_vae: str, audio_vae: str) -> dict:
    return {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"},
        },
        "video_vae": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
    }


def image_size_for(width: int, height: int, megapixels: float = 1.0) -> "tuple[int, int]":
    """動画のアスペクト比を保ったまま、参照画像に適した画素数へ拡大する。

    352x192 のような下書き解像度で参照画像まで作ると、静止画側の品質が
    落ちて参照の意味がなくなる。H3 側は参照画像を内部で縮小するので、
    アスペクト比だけ合わせて画素数は上げておく。
    """
    scale = (megapixels * 1_000_000 / (width * height)) ** 0.5
    return snap_dimension(round(width * scale)), snap_dimension(round(height * scale))


def _image_branch(prompt: dict, *, checkpoint: str, prompt_text: str, negative: str,
                  width: int, height: int, seed: int, steps: int,
                  save: bool, filename_prefix: str, prefix: str = "img_") -> str:
    """HiDream-O1 で静止画を作る部分を組み立て、IMAGE を出すノード名を返す。

    動画ワークフローに直接埋め込めるように、ノード名に接頭辞を付けて
    MiniMax-H3 側のノードと衝突しないようにしている。埋め込んだ場合は
    生成された IMAGE をそのまま first_frame / ref_images へ繋げられるので、
    中間ファイルの受け渡しが要らない（ComfyUI のキューにも1件しか積まれない）。
    """
    d = IMAGE_DEFAULTS
    n = lambda k: prefix + k  # noqa: E731

    prompt[n("ckpt")] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint},
    }
    prompt[n("model")] = {
        "class_type": "ModelNoiseScale",
        "inputs": {"model": [n("ckpt"), 0], "noise_scale": d["noise_scale"]},
    }
    prompt[n("pos")] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt_text, "clip": [n("ckpt"), 1]},
    }
    prompt[n("neg")] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": [n("ckpt"), 1]},
    }
    prompt[n("latent")] = {
        "class_type": "EmptyHiDreamO1LatentImage",
        "inputs": {"width": snap_dimension(width), "height": snap_dimension(height),
                   "batch_size": 1},
    }
    prompt[n("sigmas")] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [n("model"), 0], "scheduler": d["scheduler"],
                   "steps": steps, "denoise": 1.0},
    }
    prompt[n("sampler")] = {
        "class_type": "SamplerLCM",
        "inputs": {"s_noise": d["lcm_s_noise"], "s_noise_end": d["lcm_s_noise_end"],
                   "noise_clip_std": d["lcm_noise_clip_std"]},
    }
    prompt[n("sample")] = {
        "class_type": "SamplerCustom",
        "inputs": {
            "model": [n("model"), 0], "add_noise": True, "noise_seed": seed,
            "cfg": d["cfg"], "positive": [n("pos"), 0], "negative": [n("neg"), 0],
            "sampler": [n("sampler"), 0], "sigmas": [n("sigmas"), 0],
            "latent_image": [n("latent"), 0],
        },
    }
    prompt[n("decode")] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [n("sample"), 0], "vae": [n("ckpt"), 2]},
    }
    if save:
        prompt[n("save")] = {
            "class_type": "SaveImage",
            "inputs": {"images": [n("decode"), 0], "filename_prefix": filename_prefix},
        }
    return n("decode")


def build_t2i(
    *,
    prompt_text: str,
    width: int = 2048,
    height: int = 2048,
    seed: int,
    steps: int = IMAGE_DEFAULTS["steps"],
    negative: str = "",
    models: dict | None = None,
    filename_prefix: str = "hidream_o1/img",
) -> dict:
    """静止画のみを生成する（HiDream-O1）。"""
    m = {**DEFAULT_MODELS, **(models or {})}
    prompt: dict = {}
    _image_branch(prompt, checkpoint=m["image_checkpoint"], prompt_text=prompt_text,
                  negative=negative, width=width, height=height, seed=seed,
                  steps=steps, save=True, filename_prefix=filename_prefix)
    return prompt


# 音声を別ファイルでも出す場合の形式。SaveAudio は FLAC を書く。
AUDIO_SAVE_NODES = {
    "flac": ("SaveAudio", "flac", {}),
    "mp3": ("SaveAudioMP3", "mp3", {"quality": "V0"}),
    "opus": ("SaveAudioOpus", "opus", {"quality": "128k"}),
}


def _sampler_tail(prompt: dict, cond_node: str, steps: int, seed: int,
                  sampler: str, scheduler: str, filename_prefix: str,
                  audio: bool = True, audio_format: str | None = None,
                  shift_video: float | None = None,
                  shift_audio: float | None = None) -> dict:
    """conditioning + latent を受け取ってサンプリング〜動画保存までを繋ぐ。

    audio=False は「音声を生成しない」ではなく「出力動画に音声トラックを
    入れない」。H3 は映像と音声を単一の forward で同時に作るので、音声の
    計算自体は避けられず、省けるのは音声 VAE のデコードだけ。
    """
    # shift が指定されたときだけ SigmaShift を挟む（既定は公式テンプレートと同じ挙動）
    model_node = "unet"
    if shift_video is not None or shift_audio is not None:
        prompt["sigma_shift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["unet", 0],
                "shift_video": DEFAULT_SHIFT_VIDEO if shift_video is None else shift_video,
                "shift_audio": DEFAULT_SHIFT_AUDIO if shift_audio is None else shift_audio,
            },
        }
        model_node = "sigma_shift"

    prompt["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    prompt["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": [model_node, 0], "conditioning": [cond_node, 0]},
    }
    prompt["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    prompt["sigmas"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [model_node, 0], "scheduler": scheduler,
                   "steps": steps, "denoise": 1.0},
    }
    prompt["sample"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0],
            "latent_image": [cond_node, 1],
        },
    }
    prompt["decode_video"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sample", 0], "vae": ["video_vae", 0]},
    }
    video_inputs = {"images": ["decode_video", 0], "fps": float(FPS)}
    if audio or audio_format:
        prompt["decode_audio"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["audio_vae", 0]},
        }
        if audio:
            # CreateVideo の audio は optional。繋がなければ無音の動画になる
            video_inputs["audio"] = ["decode_audio", 0]
        if audio_format:
            cls, _ext, extra = AUDIO_SAVE_NODES[audio_format]
            prompt["save_audio"] = {
                "class_type": cls,
                "inputs": {"audio": ["decode_audio", 0],
                           "filename_prefix": filename_prefix + "_audio", **extra},
            }
    prompt["create_video"] = {"class_type": "CreateVideo", "inputs": video_inputs}
    prompt["save"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return prompt


def build_fl2va(
    *,
    prompt_text: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int = 20,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    first_frame: str | None = None,
    last_frame: str | None = None,
    prep_image: dict | None = None,
    audio: bool = True,
    audio_format: str | None = None,
    shift_video: float | None = None,
    shift_audio: float | None = None,
    models: dict | None = None,
    filename_prefix: str = "minimax_h3/h3",
) -> dict:
    """T2V / I2V / 先頭+末尾フレーム指定 (FL2VA チェックポイント)。"""
    m = {**DEFAULT_MODELS, **(models or {})}
    prompt = _loaders(m["unet_fl2va"], m["clip"], m["video_vae"], m["audio_vae"])

    cond_inputs = {
        "clip": ["clip", 0],
        "vae": ["video_vae", 0],
        "prompt": prompt_text,
        "width": snap_dimension(width),
        "height": snap_dimension(height),
        "length": length,
    }
    if first_frame:
        prompt["first_frame"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        cond_inputs["first_frame"] = ["first_frame", 0]
    elif prep_image:
        # 参照用の静止画を同じグラフ内で先に作り、その IMAGE を直接繋ぐ。
        # 中間ファイルの往復が無く、ComfyUI のキューにも1件しか積まれない。
        node = _image_branch(
            prompt, checkpoint=m["image_checkpoint"],
            prompt_text=prep_image["prompt"], negative=prep_image.get("negative", ""),
            width=prep_image["width"], height=prep_image["height"],
            seed=prep_image["seed"], steps=prep_image["steps"],
            save=prep_image.get("save", True),
            filename_prefix=filename_prefix + "_ref")
        cond_inputs["first_frame"] = [node, 0]
    if last_frame:
        prompt["last_frame"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
        cond_inputs["last_frame"] = ["last_frame", 0]

    prompt["cond"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}
    return _sampler_tail(prompt, "cond", steps, seed, sampler, scheduler, filename_prefix,
                         audio=audio, audio_format=audio_format,
                         shift_video=shift_video, shift_audio=shift_audio)


def build_ref2va(
    *,
    prompt_text: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int = 20,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    ref_images: list[str] | None = None,
    ref_image_size: str = "match",
    prep_image: dict | None = None,
    audio: bool = True,
    audio_format: str | None = None,
    shift_video: float | None = None,
    shift_audio: float | None = None,
    models: dict | None = None,
    filename_prefix: str = "minimax_h3/h3_ref",
) -> dict:
    """参照画像つき生成 (Ref2VA チェックポイント)。

    プロンプト中で <Picture 1>, <Picture 2> ... と参照する。
    """
    m = {**DEFAULT_MODELS, **(models or {})}
    prompt = _loaders(m["unet_ref2va"], m["clip"], m["video_vae"], m["audio_vae"])

    cond_inputs = {
        "clip": ["clip", 0],
        "vae": ["video_vae", 0],
        "audio_vae": ["audio_vae", 0],
        "prompt": prompt_text,
        "width": snap_dimension(width),
        "height": snap_dimension(height),
        "length": length,
        "ref_image_size": ref_image_size,
    }

    # Autogrow 入力は {"<prefix><n>": <link>} という辞書で渡す
    autogrow = {}
    slot = 0
    if prep_image:
        # 生成した静止画を <Picture 1> として先頭に置く
        node = _image_branch(
            prompt, checkpoint=m["image_checkpoint"],
            prompt_text=prep_image["prompt"], negative=prep_image.get("negative", ""),
            width=prep_image["width"], height=prep_image["height"],
            seed=prep_image["seed"], steps=prep_image["steps"],
            save=prep_image.get("save", True),
            filename_prefix=filename_prefix + "_ref")
        autogrow["ref_image_0"] = [node, 0]
        slot = 1
    for i, name in enumerate(ref_images or []):
        node_id = f"ref_image_{slot + i}"
        prompt[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        autogrow[f"ref_image_{slot + i}"] = [node_id, 0]
    if autogrow:
        cond_inputs["ref_images"] = autogrow

    prompt["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    return _sampler_tail(prompt, "cond", steps, seed, sampler, scheduler, filename_prefix,
                         audio=audio, audio_format=audio_format,
                         shift_video=shift_video, shift_audio=shift_audio)
