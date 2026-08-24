"""Shared helpers for the RefMod plugin: mod storage folder + media loading.

Media tensors are produced in the exact convention Wan2GP's MiniMax H3
pipeline uses internally (see models/minimax_h3/pipeline.py's ``_pil_to_video``
/``_as_video``): channel-first ``[C, T, H, W]``, float, values in [-1, 1].
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .core import H3RefMod, infer_kind_from_tags, read_refmod_meta

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".gif"}


def plugin_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_REFMODS_DIR_REL = ("loras", "refmods_plugin", "minimax_h3")  # nested under loras/ (a folder
                                                               # Wan2GP itself already owns and
                                                               # won't repurpose) rather than a
                                                               # top-level refmods/, which some
                                                               # future Wan2GP update could
                                                               # start using for something else.


def refmods_dir() -> str:
    """<wan2gp root>/loras/refmods_plugin/minimax_h3 -- created on first use.
    Nested under loras/ specifically so a future Wan2GP update creating its
    own top-level refmods/ folder can never collide with this plugin's data."""
    # cwd is the Wan2GP install root while the app is running (wgp.py relies
    # on the same assumption for its own "loras/" folder), so a plain
    # relative path keeps mods next to the rest of the user's data instead of
    # inside this plugin's own folder.
    d = os.path.join(os.getcwd(), *_REFMODS_DIR_REL)
    os.makedirs(d, exist_ok=True)
    return d


def mod_path(name: str) -> str:
    return os.path.join(refmods_dir(), _sanitize_name(name))


def _sanitize_name(name: str) -> str:
    name = "".join(c for c in str(name).strip() if c.isalnum() or c in ("_", "-", " ")).strip()
    name = name.replace(" ", "_")
    return name or "refmod"


def list_refmods() -> List[str]:
    """Names of every saved mod (without extension), sorted. Reads the
    metadata header only -- no tensors are loaded."""
    d = refmods_dir()
    if not os.path.isdir(d):
        return []
    names = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".safetensors"):
            names.append(fn[:-len(".safetensors")])
    return names


def list_refmods_by_kind(kind: str) -> List[str]:
    """Names (matching list_refmods()/mod_path(), *not* the metadata's own
    "name" field, so they're always safe to pass straight to load_refmod())
    of saved mods whose kind is "image" or "video". Used to pre-filter the
    Generate UI's per-kind mod pickers so a slot can only ever be pointed at
    a mod that actually fits it."""
    out = []
    for name in list_refmods():
        try:
            meta = read_refmod_meta(mod_path(name))
            if meta is not None and meta.get("kind") == kind:
                out.append(name)
        except Exception:
            pass
    return out


def list_refmods_info() -> List[dict]:
    """[{name, kind, mode, tokens, source, description, concept_type, size_mb}, ...]
    for the Library tab. Skips files that fail to parse instead of raising."""
    out = []
    for name in list_refmods():
        try:
            meta = read_refmod_meta(mod_path(name))
            if meta is None:
                continue
            latent_h = int(meta.get("latent_h", 0))
            latent_w = int(meta.get("latent_w", 0))
            latent_t = int(meta.get("latent_t", 1))
            tokens = (latent_h // 2) * (latent_w // 2) * latent_t
            size_mb = os.path.getsize(mod_path(name) + ".safetensors") / (1024 * 1024)
            out.append({
                "name": meta.get("name", name),
                "kind": meta.get("kind", "?"),
                "mode": meta.get("mode", "?"),
                "tokens": tokens,
                "source": meta.get("source", ""),
                "description": meta.get("description", ""),
                "concept_type": meta.get("concept_type", "generic"),
                "size_mb": round(size_mb, 3),
            })
        except Exception as e:
            print(f"[H3RefMod] could not read {name}: {e}")
    return out


def delete_refmod(name: str) -> bool:
    p = mod_path(name) + ".safetensors"
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def load_refmod(name: str, device: str = "cpu") -> H3RefMod:
    return H3RefMod.load(mod_path(name), device=device)


def reclassify_mod(name: str) -> Optional[bool]:
    """Fix a single mod's ``kind`` if it was mis-tagged by a pre-0.10 version
    of this plugin (which classified *any* multi-frame mod as "video", even
    one built purely by stacking several still images with zero real video
    sources). Rewrites the file in place with the same latent tensor, only
    the metadata changes. Returns True if it was fixed, False if it was
    already correct, or None if the mod's own ``tags`` don't record enough
    information to tell (very old/hand-made files -- left untouched)."""
    meta = read_refmod_meta(mod_path(name))
    if meta is None:
        return None
    current_kind = meta.get("kind", "image")
    correct_kind = infer_kind_from_tags(meta.get("tags"), fallback=current_kind)
    if correct_kind == current_kind:
        return False
    mod = load_refmod(name)
    mod.kind = correct_kind
    mod.save(mod_path(name))
    return True


def reclassify_all_mods() -> Tuple[int, int]:
    """Runs reclassify_mod() over every saved mod. Returns (fixed, checked)."""
    names = list_refmods()
    fixed = 0
    for name in names:
        try:
            if reclassify_mod(name):
                fixed += 1
        except Exception as e:
            print(f"[H3RefMod] could not check/fix classification for '{name}': {e!r}")
    return fixed, len(names)


def rename_and_update_mod(old_name: str, new_name: Optional[str] = None,
                          new_description: Optional[str] = None) -> str:
    """Rename a saved mod and/or update its description in place -- the
    latent data is untouched either way, only metadata changes (and, for a
    rename, the file name). Returns the mod's final name (same as
    ``old_name`` if no rename happened, or if the sanitized new name is
    identical to the old one). Raises ValueError if a mod already exists
    under the requested new name (never silently overwrites another mod)."""
    mod = load_refmod(old_name)
    target_name = _sanitize_name(new_name) if new_name else old_name
    if new_description is not None:
        mod.description = new_description
    if target_name != old_name:
        if os.path.isfile(mod_path(target_name) + ".safetensors"):
            raise ValueError(f"A mod named '{target_name}' already exists -- pick a different name.")
        mod.name = target_name
        mod.save(mod_path(target_name))
        delete_refmod(old_name)
    else:
        mod.name = old_name
        mod.save(mod_path(old_name))
    return target_name


# ═══════════════════════════════════════════════════════════════════════════
# Media loading -> CTHW tensors in [-1, 1]
# ═══════════════════════════════════════════════════════════════════════════


def pil_to_cthw(image: Image.Image) -> torch.Tensor:
    """PIL image -> [C, 1, H, W] float32 in [-1, 1] (matches Wan2GP's own
    ``_pil_to_video`` in models/minimax_h3/pipeline.py)."""
    image = image.convert("RGB")
    arr = np.asarray(image).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1)


def gradio_image_to_cthw(value) -> Optional[torch.Tensor]:
    """Accepts what a gr.Image/gr.Gallery entry can hand back: a file path,
    a PIL Image, or a numpy array."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, str):
        with Image.open(value) as img:
            return pil_to_cthw(img.copy())
    if isinstance(value, Image.Image):
        return pil_to_cthw(value)
    if isinstance(value, np.ndarray):
        return pil_to_cthw(Image.fromarray(value))
    raise ValueError(f"Unsupported image input type: {type(value)!r}")


def load_video_cthw(path: str, max_frames: int = 240) -> torch.Tensor:
    """Load a video file -> [C, T, H, W] float32 in [-1, 1]. Uses opencv if
    available, else imageio (same fallback chain as the ComfyUI RefMod pack)."""
    frames = None
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        targets = None
        if total > max_frames:
            targets = set(np.linspace(0, total - 1, max_frames).round().astype(int).tolist())
        out, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if targets is not None and idx not in targets:
                idx += 1
                continue
            idx += 1
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if out:
            frames = np.stack(out)
    except Exception:
        frames = None

    if frames is None:
        try:
            import imageio.v2 as imageio
            reader = imageio.get_reader(path)
            out = []
            for i, frame in enumerate(reader):
                if max_frames and i >= max_frames:
                    break
                out.append(np.asarray(frame)[..., :3])
            reader.close()
            if out:
                frames = np.stack(out)
        except Exception:
            frames = None

    if frames is None:
        raise RuntimeError(f"No video loader available for {path} (tried opencv and imageio).")

    n = frames.shape[0]
    if n > max_frames:
        idx = np.linspace(0, n - 1, max_frames).round().astype(int)
        frames = frames[idx]
    t = torch.from_numpy(frames.copy()).float().div_(127.5).sub_(1.0)  # [T, H, W, C]
    return t.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]


def resize_cthw(video: torch.Tensor, target_short_edge: int, canvas: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Downscale-only resize of a [C, T, H, W] tensor to ``target_short_edge``
    (or to an explicit ``canvas`` = (w, h) if given), snapped to a multiple of
    32 like Wan2GP's own reference-image canvas resolver."""
    c, t, h, w = video.shape
    if canvas is not None:
        tw, th = canvas
    else:
        scale = min(1.0, target_short_edge / min(h, w))
        tw, th = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    if (tw, th) == (w, h):
        return video
    return torch.nn.functional.interpolate(video.permute(1, 0, 2, 3), size=(th, tw),
                                           mode="bilinear", align_corners=False).permute(1, 0, 2, 3).contiguous()


def ensure_min_size(video: torch.Tensor, min_edge: int = 32) -> torch.Tensor:
    c, t, h, w = video.shape
    if h >= min_edge and w >= min_edge:
        return video
    scale = min_edge / min(h, w)
    tw, th = max(min_edge, round(w * scale)), max(min_edge, round(h * scale))
    return torch.nn.functional.interpolate(video.permute(1, 0, 2, 3), size=(th, tw),
                                           mode="bilinear", align_corners=False).permute(1, 0, 2, 3).contiguous()


# ═══════════════════════════════════════════════════════════════════════════
# Background removal (matches Wan2GP's own reference-image processing)
# ═══════════════════════════════════════════════════════════════════════════


def new_rembg_session():
    """A rembg session using Wan2GP's own model-cache location (avoids a
    second, differently-located U2NET download) when running inside a
    Wan2GP process; falls back to rembg's own default location otherwise
    (e.g. when unit-testing this plugin standalone)."""
    try:
        from shared.utils.utils import new_rembg_session as _wgp_new_rembg_session
        return _wgp_new_rembg_session()
    except Exception:
        from rembg import new_session
        return new_session()


def remove_background_from_image(img: Image.Image, session=None, bg_color=(255, 255, 255)) -> Image.Image:
    """Matches Wan2GP's own "Automatic Removal of Background behind People or
    Objects in Reference Images" exactly -- same rembg call and
    alpha-matting parameters as shared/utils/utils.py's
    resize_and_remove_background -- so a RefMod extracted with this on looks
    consistent with a live reference image processed the same way in the
    main Media Generator form. Only meant for still images: Wan2GP's own
    background removal only applies to reference *images*, not reference
    videos, and this plugin follows the same rule."""
    from rembg import remove
    if session is None:
        session = new_rembg_session()
    return remove(img.convert("RGB"), session=session, alpha_matting_erode_size=1,
                 alpha_matting=True, bgcolor=list(bg_color) + [0]).convert("RGB")
