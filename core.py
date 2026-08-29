"""
core.py -- "RefMod" compression/curve math for MiniMax H3 (no training required)

Ported for Wan2GP from ComfyUI-MiniMaxH3Mod (https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod,
MIT License, (c) 2026 Luisa). That project's core.py has zero ComfyUI-specific
imports -- it is pure PyTorch operating on the MiniMax H3 video VAE's raw
latent tensors -- so it is reproduced here essentially unchanged. Wan2GP's own
MiniMax H3 VAE (models/minimax_h3/video_vae.py) is the same official 24-channel
VAE, so mods produced by either implementation share the exact same latent
format and are interchangeable.

The reference-video path of MiniMax H3 works by injecting *reference tokens*
into the packed sequence: the ref is VAE-encoded, patchified, projected and
placed on the 3D RoPE grid, then every DiT block attends to it. A video ref is
expensive because it contributes thousands of tokens.

A RefMod is the same reference, compressed to a handful of tokens:

  * the ref is VAE-encoded to its full latent [1, 24, T, H, W],
  * mode="training": the latent is average-pooled to a tiny grid (default
    16x16) and a few latent frames, optionally refined with a few gradient
    steps that reconstruct the full latent (still no model weights involved),
  * mode="encode": the latent is kept at full resolution (max identity, more
    tokens).

At generation time the latent is handed back to the model through the same
native ref-conditioning path used for a live image/video reference, so it
flows through the exact same per-block attention machinery -- the only thing
that changes is the token budget.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Metadata key embedded in the safetensors header (keeps a mod in one file,
# so it can be shared/uploaded as a single artifact). Kept identical to the
# ComfyUI pack's key so mods are interchangeable between the two tools.
META_KEY = "refmod_meta"

CONCEPT_TYPES = (
    "generic",       # unspecified / mixed
    "identity",      # a specific person/character -- use mode="encode" for this
    "pose_motion",   # a pose, dance, gesture, camera move
    "clothing",      # an outfit / garment, decoupled from who's wearing it
    "background",    # environment / set / location plate
    "style",         # look/grade/animation style, not a concrete subject
)

# "concept_type" is documentation + a routing hint, NOT a hard constraint on
# the tensors -- it does not change any encoding/pooling/refinement math. It
# drives exactly two things, both ported here from the ComfyUI pack:
#   1. the identity/training-mode/low-pool warning below (extraction time),
#   2. build_prompt_hint()'s output (a manual, opt-in string you concatenate
#      onto your own prompt -- there is no CLIP-Vision / image-embedding
#      injection point on H3's Ref2VA path to hang an automatic "visual
#      clue" off of, so text typed into the prompt is the only channel that
#      actually reaches the model; nothing in a mod file is ever read by the
#      model automatically).


def identity_training_pool_warning(concept_type: str, mode: str, pool_h: int, pool_w: int) -> Optional[str]:
    """Mirrors the ComfyUI pack's extraction-time warning: pooling a
    concept_type="identity" mod down to a small grid in "training" mode
    averages away exactly the fine detail (eyes, nose, mouth proportions)
    that carries a specific face, which is the usual cause of a person
    drifting toward a generic "chubbier/older" look when the mod is used.
    Returns a warning string, or None if it doesn't apply."""
    if concept_type == "identity" and mode == "training" and max(pool_h, pool_w) < 16:
        return (f"concept_type='identity' with mode='training' at a {pool_h}x{pool_w} grid -- "
               f"pooling averages away exactly the detail that carries a face (this is almost "
               f"certainly the cause of a 'chubbier/older' drift). For a person, either switch to "
               f"mode='encode' (real identity, more tokens) or raise the pool grid toward 32x32+ "
               f"and expect it to still be a soft approximation, not a lock.")
    return None


def build_prompt_hint(mod_metas) -> str:
    """Mirrors the ComfyUI pack's loader "prompt_hint" output: merge each
    mod's concept_type + description into one prompt-ready string, e.g.
    "identity: ginger woman, tattooed neck; pose_motion: slow twirl into
    camera". Mods with no description are skipped -- a bare concept_type
    with nothing to say isn't a useful clue. Concatenate the result onto
    your own prompt (this plugin never does so automatically -- see the
    module docstring above for why).

    ``mod_metas``: an iterable of dicts with "concept_type"/"description"
    keys (e.g. straight from read_refmod_meta()) -- kept metadata-only
    (no H3RefMod/tensor loading needed) since this never touches the latent.
    """
    parts = []
    for meta in mod_metas:
        meta = meta or {}
        desc = meta.get("description")
        if desc:
            parts.append(f"{meta.get('concept_type', 'generic')}: {desc}")
    return "; ".join(parts)


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass (downsample then upsample), used as the
    weakening target for strengths below 1.0.

    A blurred copy stays on the latent manifold (smooth, plausible) while
    still discarding detail as strength drops -- mixing toward zero or toward
    random noise instead reads as a wrong texture rather than "weaker".
    """
    if z.dim() != 5:
        return z
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


def _blur_audio_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Audio analog of _blur_latent(): a heavy temporal low-pass (downsample
    then upsample along the time axis only) used as the same "weakening
    target" for strengths below 1.0, and the extrapolation direction for
    strengths above 1.0 -- same rationale as the visual version, just with
    no spatial H/W axes to blur (an audio latent is [1, C, S, T], channels x
    stereo x time, not channels x time x height x width)."""
    if z.dim() != 4:
        return z
    b, c, s, t = z.shape
    st = max(1, t // factor)
    flat = z.float().reshape(b, c * s, t)
    down = F.adaptive_avg_pool1d(flat, st)
    up = F.interpolate(down, size=t, mode="linear", align_corners=False)
    return up.reshape(b, c, s, t).to(z.dtype)


def infer_kind_from_tags(tags, fallback: str) -> str:
    """Best-effort correct ``kind`` from a mod's own ``tags`` (which record
    "{n_img} img, {n_vid} vid" at extraction time): "video" if any real video
    source went into it, "image" otherwise -- regardless of the latent's own
    frame count. Used both by fresh extractions and by the Library tab's
    "Fix classification" tool for mods saved before this distinction existed
    (see H3RefMod.__post_init__). Falls back to ``fallback`` if no matching
    tag is found."""
    for t in tags or []:
        m = re.match(r"\s*(\d+)\s*img\s*,\s*(\d+)\s*vid\s*", str(t))
        if m:
            return "video" if int(m.group(2)) > 0 else "image"
    return fallback


def read_refmod_meta(path_no_ext: str) -> Optional[Dict]:
    """Read the metadata block from a mod file without loading its tensors."""
    try:
        with safe_open(path_no_ext + ".safetensors", framework="pt") as f:
            meta = f.metadata()
        if meta and META_KEY in meta:
            return json.loads(meta[META_KEY])
    except Exception:
        pass
    jpath = path_no_ext + ".json"
    if os.path.isfile(jpath):
        try:
            with open(jpath) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Latent compression
# ═══════════════════════════════════════════════════════════════════════════

MODE_ALIASES = {"full": "encode", "pooled": "training"}


def normalize_mode(mode: str) -> str:
    """Map legacy mode names (full/pooled, from older ComfyUI mods) to the
    current ones (encode/training)."""
    return MODE_ALIASES.get(mode, mode)


def aspect_grid(pool_h: int, pool_w: int, aspect: float) -> Tuple[int, int]:
    """Even pool grid dims that match ``aspect`` (h/w) within the dial caps.

    The DiT patches each 2x2 latent cell into one token and the rope grid is
    area-normalized per axis, so a pooled grid that ignores the source aspect
    squishes the subject. The long edge follows the user's dial (max(pool_h,
    pool_w)) and the short edge is derived from the source aspect, both
    rounded to even for the 2x2 patch.
    """
    long_edge = max(pool_h, pool_w)
    if aspect >= 1.0:
        h, w = long_edge, long_edge / aspect
    else:
        w, h = long_edge, long_edge * aspect
    h = max(2, round(h / 2) * 2)
    w = max(2, round(w / 2) * 2)
    return h, w


def pool_latent(z: torch.Tensor, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor:
    """Average-pool a VAE latent [1, 24, T, H, W] down to a tiny grid."""
    if z.shape[2] == latent_t and z.shape[3] == latent_h and z.shape[4] == latent_w:
        return z
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise ValueError(f"pool_h/pool_w must be even (got {latent_h}x{latent_w})")
    pooled = F.adaptive_avg_pool3d(z.float(), (latent_t, latent_h, latent_w))
    return pooled.to(z.dtype)


def optimize_latent(
    z_small: torch.Tensor,
    z_full: torch.Tensor,
    steps: int = 150,
    lr: float = 0.02,
    device: Optional[torch.device] = None,
    progress_every: int = 0,
    progress_cb=None,
) -> torch.Tensor:
    """Model-free refinement of the compressed latent.

    Optimizes the small latent so its trilinearly upsampled reconstruction
    matches the full reference latent. Nothing but the tiny latent is
    trainable (~1-2K params) and no diffusion model is loaded.
    """
    if steps <= 0:
        return z_small
    device = device or z_full.device
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        target = z_full.clone().float().to(device)
        param = nn.Parameter(z_small.clone().float().to(device))
        opt = torch.optim.Adam([param], lr=lr)
        size = tuple(target.shape[2:])
        for i in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=size, mode="trilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[H3RefMod] identity refinement {i + 1}/{steps}")
            if progress_cb is not None and (i + 1) % max(1, steps // 20) == 0:
                progress_cb((i + 1) / steps)
        refined = param.detach().to(z_small.dtype)
    return refined


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame / per-step strength curve
# ═══════════════════════════════════════════════════════════════════════════

CURVE_DIRECTIONS = ("constant", "concept_at_start", "concept_at_middle",
                    "concept_at_end", "concept_at_ends")
CURVE_SHAPES = ("linear", "ease", "sigmoid", "tanh", "quadratic", "cubic",
                "exponential", "stair", "elastic", "bump", "dip")


def _ease(shape: str, x: float) -> float:
    if shape == "linear":
        return x
    if shape == "ease":
        return x * x * (3.0 - 2.0 * x)
    if shape == "sigmoid":
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    if shape == "tanh":
        return 0.5 * (math.tanh(8.0 * (x - 0.5)) + 1.0)
    if shape == "quadratic":
        return x * x
    if shape == "cubic":
        return x * x * x
    if shape == "exponential":
        return 2.0 ** x - 1.0
    if shape == "stair":
        return min(1.0, math.floor(x * 4) / 3.0)
    if shape == "elastic":
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "dip":
        return abs(2.0 * x - 1.0)
    return x


def curve_strengths(spec, t: int) -> Optional[List[float]]:
    """Resolve a curve spec to ``t`` per-frame strength multipliers in [0, 1].

    Accepts a ``(direction, shape, value)`` tuple/list (see CURVE_DIRECTIONS /
    CURVE_SHAPES). Returns None for flat/no curve so the caller keeps its
    single-strength path unchanged.
    """
    if t <= 1 or spec is None or spec == "":
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 3 and isinstance(spec[0], str) and isinstance(spec[1], str):
        direction, shape, value = spec
        value = float(value)
        if direction == "constant":
            if shape == "linear":
                return None if value >= 1.0 else [value] * t
            p = [_ease(shape, i / (t - 1)) for i in range(t)]
            return [max(0.0, min(1.0, value * y)) for y in p]
        p = [_ease(shape, i / (t - 1)) for i in range(t)]
        if direction == "concept_at_start":
            return [max(0.0, min(1.0, value * y)) for y in p]
        if direction == "concept_at_end":
            return [max(0.0, min(1.0, value * (1.0 - y))) for y in p]
        if direction == "concept_at_middle":
            return [max(0.0, min(1.0, value * (1.0 - abs(2.0 * y - 1.0)))) for y in p]
        if direction == "concept_at_ends":
            return [max(0.0, min(1.0, value * abs(2.0 * y - 1.0))) for y in p]
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Token budget cap / temporal dedup (shared by the Extract step)
# ═══════════════════════════════════════════════════════════════════════════


def snap_to_causal_grid(n_frames: int) -> int:
    """Round a pixel-frame count down to the nearest valid ``4k + 1``.

    MiniMax H3's video VAE is causal: it compresses time in groups of 4 with
    one leading keyframe, so it only accepts pixel-frame counts of the form
    4k+1 (1, 5, 9, 13, 17, ...). The official reference path already trims to
    this grid before encoding.
    """
    if n_frames <= 1:
        return 1
    return ((n_frames - 1) // 4) * 4 + 1


def dedup_frame_indices(z: torch.Tensor, threshold: float = 0.02) -> List[int]:
    """Indices of latent frames kept by greedy temporal dedup."""
    t = z.shape[2]
    if t <= 1:
        return list(range(t))
    flat = z[0].float()
    kept = [0]
    prev = flat[:, 0]
    for i in range(1, t):
        cur = flat[:, i]
        denom = (cur.abs().mean() + prev.abs().mean()) / 2 + 1e-6
        diff = (cur - prev).abs().mean() / denom
        if diff >= threshold:
            kept.append(i)
            prev = cur
    return kept


def fit_token_budget(latent: torch.Tensor, budget: int, label: str) -> Tuple[torch.Tensor, List[str]]:
    """Bring a stacked latent's injected token count under ``budget``.
    Returns (latent, messages) -- messages is a list of human-readable
    strings describing what was trimmed, if anything (empty if nothing was
    over budget). The caller is responsible for surfacing these to the user
    (this function has no UI/status callback of its own)."""
    messages: List[str] = []
    h, w = latent.shape[3], latent.shape[4]
    per_frame = (h // 2) * (w // 2)
    t = latent.shape[2]
    if per_frame * t <= budget:
        return latent, messages
    kept = dedup_frame_indices(latent)
    if len(kept) < t:
        messages.append(f"{label}: over {budget}-token cap, dropped {t - len(kept)} "
                        f"near-duplicate frame(s) ({t} -> {len(kept)})")
        latent = latent[:, :, kept]
        t = latent.shape[2]
    if per_frame * t > budget:
        fit_t = max(1, budget // per_frame)
        if fit_t < t:
            idx = torch.linspace(0, t - 1, fit_t, device=latent.device).round().long()
            latent = latent[:, :, idx]
            messages.append(f"{label}: still over {budget}-token cap, resampled {t} -> {fit_t} frame(s)")
    return latent, messages


def dedup_audio_frame_indices(z: torch.Tensor, threshold: float = 0.02) -> List[int]:
    """Audio analog of dedup_frame_indices() -- greedy temporal dedup along
    an audio latent's own time axis (dim -1 of a [1, C, S, T] tensor,
    versus dim 2 of a visual [1, C, T, H, W] tensor)."""
    t = z.shape[-1]
    if t <= 1:
        return list(range(t))
    flat = z[0].float().reshape(-1, t)  # [C*S, T]
    kept = [0]
    prev = flat[:, 0]
    for i in range(1, t):
        cur = flat[:, i]
        denom = (cur.abs().mean() + prev.abs().mean()) / 2 + 1e-6
        diff = (cur - prev).abs().mean() / denom
        if diff >= threshold:
            kept.append(i)
            prev = cur
    return kept


def fit_audio_token_budget(latent: torch.Tensor, budget: int, label: str) -> Tuple[torch.Tensor, List[str]]:
    """Audio analog of fit_token_budget(). An audio-kind ref's row/token
    count in the packed sequence is latent_t * MINIMAX_H3_AUDIO_CHANNELS (2,
    stereo) -- see Wan2GP's own components/packing.py
    (``ref.num_audio_rows = ref.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS``),
    not the 2x2-patch spatial formula visual refs use."""
    messages: List[str] = []
    per_frame = 2  # MINIMAX_H3_AUDIO_CHANNELS -- fixed by the audio VAE/packing format, not user-configurable
    t = latent.shape[-1]
    if per_frame * t <= budget:
        return latent, messages
    kept = dedup_audio_frame_indices(latent)
    if len(kept) < t:
        messages.append(f"{label}: over {budget}-token cap, dropped {t - len(kept)} "
                        f"near-duplicate audio frame(s) ({t} -> {len(kept)})")
        latent = latent[..., kept]
        t = latent.shape[-1]
    if per_frame * t > budget:
        fit_t = max(1, budget // per_frame)
        if fit_t < t:
            idx = torch.linspace(0, t - 1, fit_t, device=latent.device).round().long()
            latent = latent[..., idx]
            messages.append(f"{label}: still over {budget}-token cap, resampled {t} -> {fit_t} audio frame(s)")
    return latent, messages


@dataclass
class H3RefMod:
    """A compressed reference for MiniMax H3.

    ``latent`` is the VAE latent -- for ``kind in ("image", "video")``,
    ``[1, 24, latent_t, latent_h, latent_w]``, a full-resolution encode
    (mode="encode") or a pooled thumbnail (mode="training"); for
    ``kind == "audio"``, ``[1, 32, 2, latent_t]`` (channels x stereo x
    time), always a full-resolution encode -- there is no spatial grid to
    pool for audio, so "training" mode's compression concept doesn't apply
    and audio mods are always extracted at full VAE fidelity.
    ``kind`` is "video" if any real video source was included when the mod
    was extracted, "image" otherwise -- an "image" mod can still have
    ``latent_t > 1`` if several still images were stacked into it (each
    occupies its own image-reference slot at injection time, matching
    Wan2GP's own ``refs`` payload kinds). "audio" mods are always extracted
    from a single audio file, never combined with image/video sources (the
    latent shapes are structurally incompatible to stack together).
    """

    name: str
    kind: str
    latent: torch.Tensor
    latent_h: int = 4
    latent_w: int = 4
    latent_t: int = 1
    mode: str = "training"
    source: str = ""
    source_shape: str = ""
    pool: str = ""
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    concept_type: str = "generic"

    def __post_init__(self):
        if self.kind not in ("image", "video", "audio"):
            raise ValueError(f"kind must be 'image', 'video', or 'audio' (got {self.kind!r})")
        # NOTE: kind=="image" does NOT force latent_t=1 -- an "image" mod can be
        # a stack of several independent still images (extracted together into
        # one file), each still occupying its own single-frame image-reference
        # slot at injection time (see patches.py's _inject_refmods, which
        # splits a multi-frame "image" mod back into one sentinel per frame).
        # "kind" reflects whether any *real* video source went into the mod,
        # not the frame count -- three unrelated photos stacked together are
        # still three images, not a 3-frame video, even though the underlying
        # latent tensor happens to have 3 frames either way.

    @property
    def token_count(self) -> int:
        if self.kind == "audio":
            # ref.num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS (2) --
            # see Wan2GP's own components/packing.py. Not the 2x2-patch spatial
            # formula visual refs use; audio has no spatial grid to patchify.
            return self.latent_t * 2
        per_frame = (self.latent_h // 2) * (self.latent_w // 2)
        return self.latent_t * per_frame

    def weighted_latent(self, strength: float = 1.0, curve=None) -> Optional[torch.Tensor]:
        """The latent to inject, blended with a blurred copy of itself by
        ``strength`` (and optionally a per-frame ``curve``). Returns None if
        the effective strength is 0 (mod should be dropped entirely).

        Below 1.0 this weakens the mod toward the blurred copy, same as
        always. Above 1.0 the same linear blend keeps extrapolating past the
        mod's own saved latent, in the opposite direction from the blur --
        exaggerating whatever separates it from a heavily softened version
        of itself (fine detail, sharpness) rather than adding information
        that was never encoded. Treat values above 1.0 as an experiment, not
        a guaranteed "stronger" result -- there's no ceiling built in here
        beyond whatever the caller passes.
        """
        if strength <= 0.0:
            return None
        latent = self.latent
        blur_fn = _blur_audio_latent if self.kind == "audio" else _blur_latent
        if curve is not None and self.latent_t > 1:
            strengths = curve_strengths(curve, self.latent_t)
            if strengths is not None:
                t = self.latent_t
                st = torch.tensor([max(0.0, strength * s) for s in strengths],
                                  dtype=torch.float32, device=latent.device)
                st = st.view(1, 1, 1, t) if self.kind == "audio" else st.view(1, 1, t, 1, 1)
                blurred = blur_fn(latent)
                return (st * latent.float() + (1.0 - st) * blurred.float()).to(latent.dtype)
        if strength == 1.0:
            return latent
        blurred = blur_fn(latent)
        return (strength * latent.float() + (1.0 - strength) * blurred.float()).to(latent.dtype)

    def save(self, path_no_ext: str) -> str:
        os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
        meta = {
            "name": self.name, "kind": self.kind, "latent_h": self.latent_h,
            "latent_w": self.latent_w, "latent_t": self.latent_t, "mode": self.mode,
            "source": self.source, "source_shape": self.source_shape, "pool": self.pool,
            "optimize_steps": self.optimize_steps, "tags": self.tags,
            "description": self.description, "concept_type": self.concept_type,
            "_format_version": 2, "_produced_by": "wan2gp-minimax-h3-refmod",
        }
        save_file({"latent": self.latent.contiguous()}, path_no_ext + ".safetensors",
                  metadata={META_KEY: json.dumps(meta)})
        return path_no_ext + ".safetensors"

    @classmethod
    def load(cls, path_no_ext: str, device: str = "cpu") -> "H3RefMod":
        meta = read_refmod_meta(path_no_ext)
        if meta is None:
            raise ValueError(f"{path_no_ext}.safetensors has no RefMod metadata.")
        latent = load_file(path_no_ext + ".safetensors", device=device)["latent"].clone()
        # NOTE: dict.get(key, fallback) always evaluates `fallback` eagerly, even
        # when `key` is present and the fallback goes unused -- so the fallback
        # expression itself must never index a dimension that might not exist.
        # An audio latent is 4D ([1, C, S, T]); a visual one is 5D
        # ([1, C, T, H, W]); latent.shape[4] would raise on the former
        # regardless of whether meta actually had "latent_w" saved.
        if latent.dim() == 4:
            default_h, default_w, default_t = 4, 4, latent.shape[-1]
        else:
            default_h, default_w, default_t = latent.shape[3], latent.shape[4], latent.shape[2]
        return cls(
            name=meta.get("name", os.path.basename(path_no_ext)),
            kind=meta.get("kind", "image"),
            latent=latent,
            latent_h=int(meta.get("latent_h", default_h)),
            latent_w=int(meta.get("latent_w", default_w)),
            latent_t=int(meta.get("latent_t", default_t)),
            mode=normalize_mode(meta.get("mode", "training")),
            source=meta.get("source", ""),
            source_shape=meta.get("source_shape", ""),
            pool=meta.get("pool", ""),
            optimize_steps=int(meta.get("optimize_steps", 0)),
            tags=list(meta.get("tags", [])),
            description=str(meta.get("description", "") or ""),
            concept_type=str(meta.get("concept_type", "generic") or "generic"),
        )
