"""
patches.py -- wires the RefMod mechanism into Wan2GP's MiniMax H3 pipeline.

Why monkeypatching, and why *these* three methods
---------------------------------------------------
Wan2GP's own reference-conditioning code (models/minimax_h3/pipeline.py) builds
two parallel lists inside ``MiniMaxH3Pipeline.generate()``: ``refs`` (kind /
shape metadata) and ``visual_latents`` (the actual VAE latents). Both are
local variables assembled by ``_add_image_reference`` / ``_add_video_reference``
and are only "closed off" into the packed conditioning right before sampling
starts. A plugin living outside wgp.py's source tree cannot reach into those
locals -- so instead of re-implementing (and fatally risking drifting out of
sync with) the ~300-line ``generate()`` method, this module:

1. Wraps ``_add_image_reference`` / ``_add_video_reference`` so that, when
   handed one of our own lightweight sentinel objects instead of a real
   image/video tensor, they append the *precomputed* RefMod latent straight
   into ``refs`` / ``visual_latents`` -- skipping the pixel resize + VAE
   encode, but otherwise going through the exact same, unmodified code path
   a live reference would. This is the entire point of a "RefMod": skip the
   repeated encode, not the model's own attention mechanism.

2. Wraps ``generate()`` itself so that, right before calling the original,
   it reads a small JSON blob out of the ``custom_settings`` dict (Wan2GP's
   existing generic per-model settings channel, already plumbed end to end
   from a submitted task to ``pipeline.generate(**kwargs)``) and turns it
   into sentinel objects appended to ``input_ref_images`` / ``input_frames``
   / ``input_frames2`` -- the same public parameters a live reference would
   use. This means RefMods work even when the user supplies *no* live
   reference at all, which is the main point of the feature.

3. The same ``generate()`` wrapper also recognizes a second, distinct
   ``custom_settings`` key that means "this call is a RefMod *extraction*
   job, not a real render": it runs the VAE encode + optional compression
   directly (a few seconds of work), saves the .safetensors file, and
   returns ``None`` immediately -- the same graceful no-output outcome
   Wan2GP already produces when a user aborts a generation, so nothing
   downstream needs to change to handle it.

4. Declares our two ``custom_settings`` ids on MiniMax H3's model
   definition (see ``_install_model_def_patch`` below). This one is not
   optional: Wan2GP validates *every* submitted task's ``custom_settings``
   against the ids the target model declares, and silently drops anything
   else -- so without this, points 2 and 3 above would never actually see
   our payload, with no error anywhere to explain why.

None of this touches Wan2GP's own files on disk; it is applied at import
time to the already-loaded ``MiniMaxH3Pipeline`` class, and is idempotent
(safe to call ``install_patches()`` more than once).
"""

from __future__ import annotations

import functools
import json
import random
import traceback
from typing import Optional

import torch

from . import core, storage

SETTING_GENERATE = "h3_refmod_state"     # custom_settings key: mods to inject into a real render
SETTING_EXTRACT = "h3_refmod_extract"    # custom_settings key: "run an extraction, not a render"
STASH_KEY = "_h3refmod_selection"        # key inside the session `state` dict for the inline panel
FPS_ASSUMED_FOR_DURATION_ESTIMATE = 24   # matches plugin.py's own constant of the same name --
                                         # MiniMax H3's own default fps, used only to turn a
                                         # latent frame count into an estimated-seconds figure
                                         # for status messages.

_PATCH_MARKER = "_h3refmod_plugin_patched"


class _RefModImageSentinel:
    """Stands in for a single-frame ("image kind") reference. Carries an
    already-encoded, already-weighted VAE latent [1, 24, 1, H, W]."""
    __slots__ = ("latent",)

    def __init__(self, latent: torch.Tensor):
        self.latent = latent


class _RefModVideoSentinel:
    """Stands in for a multi-frame ("video kind") reference. Carries an
    already-encoded, already-weighted VAE latent [1, 24, T, H, W].

    ``generate()`` touches ``input_frames``/``input_frames2`` more than once
    before ever reaching ``_add_video_reference`` (which is the only place
    that's patched to actually recognize this sentinel and use its latent):
    it runs every entry through ``_as_video()`` first (also patched, to pass
    a sentinel through untouched), then computes a total-duration budget
    check via ``sum(video.shape[1] for video in video_sources) / fps``. The
    ``.shape`` property below exists purely to survive *that* second access
    without crashing -- it's a plausible reconstructed pixel-space
    [C, T, H, W] shape derived from the latent's own dims (undoing the video
    VAE's causal 4:1 temporal compression and 16x spatial downsampling), not
    real pixel data (there is none for a RefMod)."""
    __slots__ = ("latent",)

    def __init__(self, latent: torch.Tensor):
        self.latent = latent

    @property
    def shape(self):
        t = self.latent.shape[2]
        t_px = (t - 1) * 4 + 1 if t > 1 else 1
        h_px = self.latent.shape[3] * 16
        w_px = self.latent.shape[4] * 16
        return (3, t_px, h_px, w_px)


def _log(msg: str) -> None:
    print(f"[H3RefMod] {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_DEF_PATCH_MARKER = "_h3refmod_model_def_patched"


def _count_refmod_visual_refs(state_json: str) -> int:
    """How many "visual reference" slots the RefMods named in a
    SETTING_GENERATE payload will occupy once injected -- mirrors
    _inject_refmods()'s own counting exactly: an image-kind mod contributes
    one slot per stacked frame times its copies (each frame becomes its own
    image reference, see core.H3RefMod's docstring), a video-kind mod
    contributes exactly one slot regardless of copies (copies repeats its
    frames within that one slot, doesn't spawn extra ones). Metadata-only
    (no tensor loading) since this only needs kind/latent_t."""
    try:
        state = json.loads(state_json)
    except Exception:
        return 0
    total = 0
    for row in state.get("rows") or []:
        name = row.get("mod")
        strength = float(row.get("strength", 1.0) or 0)
        if not name or strength <= 0:
            continue
        copies = max(1, min(10, int(row.get("copies", 1))))
        meta = storage.read_refmod_meta(storage.mod_path(name))
        if meta is None:
            continue
        if meta.get("kind", "image") == "image":
            total += max(1, int(meta.get("latent_t", 1))) * copies
        else:
            total += 1
    return total


_VALIDATE_PATCH_MARKER = "_h3refmod_validate_patched"
_INSUFFICIENT_VISUAL_MARKER = "at least as many reference images and videos as audio references"


def _install_model_def_patch() -> None:
    """Declare our two custom_settings IDs on MiniMax H3's model definition.

    Without this, Wan2GP's own task-submission validation
    (``collect_custom_settings_from_inputs``, called from ``validate_settings``
    for *every* task, including ones submitted through the API/plugin path)
    only keeps ``custom_settings`` entries whose id is declared in the
    model's own ``model_def["custom_settings"]`` list -- anything else is
    silently replaced with ``None`` before the task ever reaches
    ``pipeline.generate()``. MiniMax H3 doesn't declare any custom settings
    of its own, so without this patch our RefMod payload never survives the
    trip from a submitted task to ``generate()``'s ``kwargs`` (it *looks*
    like it worked -- no error anywhere -- generate() just silently runs a
    completely normal render instead of seeing our payload).

    This adds two plain "text" custom settings (ids ``h3_refmod_state`` /
    ``h3_refmod_extract``, matching the SETTING_GENERATE / SETTING_EXTRACT
    keys above) to MiniMax H3 Ref2VA's model definition. They will show up
    as two extra (normally empty) text boxes under "Custom Settings" on
    Wan2GP's own Media Generator form for this model -- harmless, and not
    meant to be hand-edited there; this plugin's own Generate/Extract
    buttons fill them in on submission.
    """
    try:
        from models.minimax_h3 import minimax_h3_handler
    except Exception as e:
        _log(f"could not import minimax_h3_handler to declare custom settings ({e!r}); "
             f"RefMod extraction/injection will likely silently no-op instead of taking effect.")
        return

    FamilyHandler = minimax_h3_handler.family_handler
    if getattr(FamilyHandler, _MODEL_DEF_PATCH_MARKER, False):
        return

    _orig_query_model_def = FamilyHandler.query_model_def

    extra_settings = [
        {"id": SETTING_GENERATE, "name": "H3RefModState",
         "label": "RefMods selection (managed by the MiniMax H3 RefMods plugin -- leave blank)",
         "type": "text", "default": ""},
        {"id": SETTING_EXTRACT, "name": "H3RefModExtract",
         "label": "RefMod extraction job (managed by the MiniMax H3 RefMods plugin -- leave blank)",
         "type": "text", "default": ""},
    ]

    @staticmethod
    def patched_query_model_def(base_model_type, model_def):
        result = _orig_query_model_def(base_model_type, model_def)
        if isinstance(result, dict):
            existing = result.get("custom_settings")
            existing = list(existing) if isinstance(existing, list) else []
            existing_ids = {e.get("id") for e in existing if isinstance(e, dict)}
            result["custom_settings"] = existing + [s for s in extra_settings if s["id"] not in existing_ids]
        return result

    FamilyHandler.query_model_def = patched_query_model_def
    setattr(FamilyHandler, _MODEL_DEF_PATCH_MARKER, True)
    _log("declared h3_refmod_state / h3_refmod_extract custom settings on MiniMax H3's model definition")

    if getattr(FamilyHandler, _VALIDATE_PATCH_MARKER, False):
        return
    _orig_validate = getattr(FamilyHandler, "validate_generative_settings", None)
    if _orig_validate is None:
        return

    @staticmethod
    def patched_validate_generative_settings(base_model_type, model_def, inputs):
        error = _orig_validate(base_model_type, model_def, inputs)
        # Wan2GP's own pre-flight check (called before pipeline.generate() ever
        # runs) only counts *native* image_refs/video_guide fields -- it has no
        # visibility into RefMods, which only turn into visual references much
        # later, inside our own generate() patch. Rather than re-implementing
        # every rule this function enforces (durations, per-type caps, control-
        # video-specific checks...), only step in for this one specific failure:
        # if RefMods would supply enough visual references to satisfy it, clear
        # it; every other check the original function performs is untouched.
        if error and _INSUFFICIENT_VISUAL_MARKER in error:
            custom_settings = inputs.get("custom_settings")
            state_json = custom_settings.get(SETTING_GENERATE) if isinstance(custom_settings, dict) else None
            if state_json:
                refmod_visual_count = _count_refmod_visual_refs(state_json)
                if refmod_visual_count > 0:
                    video_prompt_type = str(inputs.get("video_prompt_type") or "")
                    audio_prompt_type = str(inputs.get("audio_prompt_type") or "")
                    image_count = len(inputs.get("image_refs") or [])
                    video_count = (1 if "V" in video_prompt_type else 0) + (1 if "+" in video_prompt_type else 0)
                    audio_count = (1 if "A" in audio_prompt_type else 0) + (1 if "B" in audio_prompt_type else 0)
                    visual_count = image_count + video_count + refmod_visual_count
                    if audio_count <= visual_count:
                        _log(f"RefMods supply {refmod_visual_count} visual reference(s) -- "
                             f"clearing the native '{audio_count} audio vs "
                             f"{image_count + video_count} visual' pre-flight check "
                             f"({visual_count} visual once RefMods are counted).")
                        return None
        return error

    FamilyHandler.validate_generative_settings = patched_validate_generative_settings
    setattr(FamilyHandler, _VALIDATE_PATCH_MARKER, True)
    _log("patched validate_generative_settings so RefMods count as visual references "
         "against MiniMax H3's audio-reference pre-flight check")


def install_patches() -> Optional[str]:
    """Apply the monkeypatches. Returns None on success, or an error string
    (also printed) if Wan2GP's MiniMax H3 module could not be found -- e.g.
    a very different Wan2GP version. Safe to call more than once."""
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
    except Exception as e:
        msg = (f"could not import models.minimax_h3.pipeline ({e!r}); this Wan2GP "
               f"install may not include MiniMax H3, or its layout has changed. "
               f"RefMod extraction/injection will not be available.")
        _log(msg)
        return msg

    _install_model_def_patch()

    Pipeline = h3_pipeline.MiniMaxH3Pipeline
    if getattr(Pipeline, _PATCH_MARKER, False):
        return None  # already patched (e.g. plugin reloaded)

    _orig_add_image_reference = Pipeline._add_image_reference
    _orig_add_video_reference = Pipeline._add_video_reference
    _orig_generate = Pipeline.generate
    _orig_as_video = getattr(h3_pipeline, "_as_video", None)

    @functools.wraps(_orig_add_image_reference)
    def patched_add_image_reference(self, image, target_width, target_height,
                                     image_refs_relative_size, presentation, visual_latents, refs):
        if isinstance(image, _RefModImageSentinel):
            latent = image.latent
            visual_latents.append(latent)
            refs.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})
            return
        return _orig_add_image_reference(self, image, target_width, target_height,
                                          image_refs_relative_size, presentation, visual_latents, refs)

    @functools.wraps(_orig_add_video_reference)
    def patched_add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        if isinstance(video, _RefModVideoSentinel):
            latent = video.latent
            visual_latents.append(latent)
            refs.append({"kind": "video", "latent_t": latent.shape[2],
                         "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "ref_audio_t": 0})
            return
        return _orig_add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs)

    if _orig_as_video is not None:
        @functools.wraps(_orig_as_video)
        def patched_as_video(source):
            # generate() runs every entry of `video_sources` (= [input_frames,
            # input_frames2]) through _as_video() itself, *before* looping over
            # them to call _add_video_reference -- so our sentinel has to survive
            # this call too, not just the one inside _add_video_reference, or it
            # crashes here first with "'_RefModVideoSentinel' object has no
            # attribute 'ndim'" before our other patch ever gets a chance to run.
            if isinstance(source, _RefModVideoSentinel):
                return source
            return _orig_as_video(source)
        h3_pipeline._as_video = patched_as_video
    else:
        _log("could not find _as_video in models.minimax_h3.pipeline to patch; "
             "video-kind RefMods will likely fail with an AttributeError at generation time. "
             "Image-kind RefMods and extraction are unaffected.")

    _orig_resize_video = getattr(h3_pipeline, "_resize_video", None)
    if _orig_resize_video is not None:
        @functools.wraps(_orig_resize_video)
        def patched_resize_video(video, height, width):
            # Newer Wan2GP builds resize each reference video to the output
            # resolution right before handing it to _add_video_reference. A
            # RefMod sentinel carries an already-VAE-encoded latent, not pixels
            # -- there is nothing meaningful to bicubic-resize (and no pixel
            # tensor to .permute()), so it passes through untouched and the
            # latent goes into the packed sequence at its own saved
            # resolution, exactly as it did before this step existed.
            if isinstance(video, _RefModVideoSentinel):
                return video
            return _orig_resize_video(video, height, width)
        h3_pipeline._resize_video = patched_resize_video
    else:
        _log("no _resize_video found in models.minimax_h3.pipeline -- fine on older Wan2GP "
             "builds that don't have it; on newer ones video-kind RefMods would fail with "
             "\"'_RefModVideoSentinel' object has no attribute 'permute'\".")

    @functools.wraps(_orig_generate)
    def patched_generate(self, *args, **kwargs):
        custom_settings = kwargs.get("custom_settings")
        custom_settings = custom_settings if isinstance(custom_settings, dict) else {}

        extract_job = custom_settings.get(SETTING_EXTRACT)
        if extract_job:
            set_progress_status = kwargs.get("set_progress_status")
            try:
                _run_extract_job(self, extract_job, set_progress_status=set_progress_status)
            except Exception:
                _log("extraction job failed:\n" + traceback.format_exc())
                if set_progress_status is not None:
                    try:
                        set_progress_status("RefMod extraction failed -- see the console/log for details")
                    except Exception:
                        pass
            return None  # graceful no-output outcome, same as a user-initiated abort

        state_json = custom_settings.get(SETTING_GENERATE)
        if state_json:
            try:
                kwargs = _inject_refmods(self, kwargs, state_json)
            except Exception:
                _log("could not inject RefMods, continuing without them:\n" + traceback.format_exc())

        return _orig_generate(self, *args, **kwargs)

    Pipeline._add_image_reference = patched_add_image_reference
    Pipeline._add_video_reference = patched_add_video_reference
    Pipeline.generate = patched_generate
    setattr(Pipeline, _PATCH_MARKER, True)
    _log("patches installed on MiniMaxH3Pipeline (RefMod extraction + injection enabled)")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inline Media Generator panel support
# ═══════════════════════════════════════════════════════════════════════════

_PREPARE_INPUTS_PATCH_MARKER = "_h3refmod_prepare_inputs_patched"


def install_prepare_inputs_dict_patch(orig_prepare_inputs_dict, get_state_model_type_fn, set_global_fn) -> Optional[str]:
    """Lets the inline RefMods panel (injected onto Wan2GP's own Media
    Generator page, see plugin.py's ``_build_inline_refmods_section``) affect
    generations started from *that page's own* Generate button, without
    duplicating a single one of its fields.

    The inline panel writes the user's RefMod selection into
    ``state[STASH_KEY]`` -- a namespaced key on the session state dict this
    plugin owns, untouched by anything else in Wan2GP. It's stored there
    (rather than directly into the model's settings dict) because *that*
    dict gets rebuilt from the live form fields on every edit (Wan2GP
    autosaves the form continuously via ``save_inputs``/``prepare_inputs_dict
    (target="state")``), which would silently wipe a one-off injection the
    next time the user touched an unrelated field like the prompt.

    This wraps ``prepare_inputs_dict`` (a plain function in wgp.py, patched
    here via ``set_global`` -- Wan2GP's own supported way for a plugin to
    replace one of its globals) so that, on every call, it re-reads
    ``state[STASH_KEY]`` and folds it into that call's ``custom_settings``
    -- surviving exactly the autosave cycle that would otherwise erase it.
    The original function's behavior is fully preserved for every other
    model, and for MiniMax H3 whenever the panel's selection is empty.
    """
    if getattr(install_prepare_inputs_dict_patch, _PREPARE_INPUTS_PATCH_MARKER, False):
        return None

    def patched_prepare_inputs_dict(target, inputs, model_type=None, model_filename=None):
        state = inputs.get("state") if isinstance(inputs, dict) else None
        result = orig_prepare_inputs_dict(target, inputs, model_type, model_filename)
        try:
            if isinstance(state, dict) and isinstance(result, dict):
                resolved_type = model_type or get_state_model_type_fn(state)
                if str(resolved_type or "").startswith("minimax_h3_ref2va"):
                    stash = state.get(STASH_KEY)
                    existing = dict(result.get("custom_settings") or {})
                    if stash:
                        existing[SETTING_GENERATE] = stash
                        result["custom_settings"] = existing
                    elif SETTING_GENERATE in existing:
                        existing.pop(SETTING_GENERATE, None)
                        result["custom_settings"] = existing or None
        except Exception:
            _log("prepare_inputs_dict patch: RefMod injection failed, leaving settings "
                 "untouched for this call:\n" + traceback.format_exc())
        return result

    try:
        set_global_fn("prepare_inputs_dict", patched_prepare_inputs_dict)
    except Exception as e:
        msg = f"could not patch prepare_inputs_dict ({e!r}); the inline Media Generator panel will not work."
        _log(msg)
        return msg
    setattr(install_prepare_inputs_dict_patch, _PREPARE_INPUTS_PATCH_MARKER, True)
    _log("hooked prepare_inputs_dict so the inline RefMods panel on the Media Generator page "
         "persists across form edits")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Generation-time injection
# ═══════════════════════════════════════════════════════════════════════════

def _inject_refmods(pipeline_self, kwargs: dict, state_json: str) -> dict:
    try:
        state = json.loads(state_json)
    except Exception as e:
        _log(f"could not parse RefMod state ({e!r}); ignoring")
        return kwargs
    rows = state.get("rows") or []
    if not rows:
        return kwargs
    retention = float(state.get("retention", 1.0))
    curve = state.get("curve")  # [direction, shape, value] or None
    if isinstance(curve, list):
        curve = tuple(curve)
    seed = int(state.get("scramble_seed", -1))

    loaded_rows = []  # (mod, strength, copies), one entry per selected row, in order
    for row in rows:
        name = row.get("mod")
        strength = float(row.get("strength", 1.0))
        copies = max(1, min(10, int(row.get("copies", 1))))
        if not name or strength <= 0.0:
            continue
        try:
            mod = storage.load_refmod(name)
        except Exception as e:
            _log(f"could not load mod {name!r}, skipping ({e!r})")
            continue
        loaded_rows.append((mod, strength, copies))

    if not loaded_rows:
        return kwargs

    if seed >= 0 and len(loaded_rows) > 1:
        rng = random.Random(seed)
        rng.shuffle(loaded_rows)
        keep = rng.randint(max(1, len(loaded_rows) // 2), len(loaded_rows))
        loaded_rows = loaded_rows[:keep]
        _log(f"scramble seed={seed}: kept {len(loaded_rows)}/{len(rows)} row(s), order shuffled")

    image_sentinels, video_sentinels = [], []
    total_tokens = 0
    for mod, strength, copies in loaded_rows:
        eff = max(0.0, min(2.0, strength * retention))
        latent = mod.weighted_latent(eff, curve=curve)
        if latent is None:
            continue
        total_tokens += mod.token_count
        if mod.kind == "image":
            # An "image" mod can still have more than one latent frame if
            # several still images were stacked together at extraction time
            # (see core.H3RefMod's docstring) -- each frame is an
            # independent image reference, so split it back into one
            # sentinel per frame here rather than sending Wan2GP's
            # single-frame-only image-reference path a multi-frame latent.
            # "Copies" duplicates the whole row's worth of image sentinels.
            t = latent.shape[2]
            for _ in range(copies):
                for j in range(t):
                    image_sentinels.append(_RefModImageSentinel(latent[:, :, j:j + 1]))
        else:
            # Video-kind mods go through Wan2GP's own native reference-video
            # slots directly -- the exact same mechanism "Use Two Reference
            # Videos" uses, one mod per slot -- rather than being merged
            # into a single combined tensor. "Copies" here repeats this
            # mod's own frames within its one slot (extending its own
            # duration) instead of spawning extra slot-consuming instances,
            # since MiniMax H3 only exposes 2 such slots in total.
            if copies > 1:
                latent = latent.repeat(1, 1, copies, 1, 1)
            video_sentinels.append(_RefModVideoSentinel(latent))

    if not image_sentinels and not video_sentinels:
        return kwargs

    if image_sentinels:
        existing = kwargs.get("input_ref_images") or []
        kwargs["input_ref_images"] = list(existing) + image_sentinels

    if video_sentinels:
        video_prompt_type = str(kwargs.get("video_prompt_type") or "")
        remaining = list(video_sentinels)
        if remaining and kwargs.get("input_frames") is None:
            kwargs["input_frames"] = remaining.pop(0)
            if "V" not in video_prompt_type:
                video_prompt_type += "V"
        if remaining and kwargs.get("input_frames2") is None:
            kwargs["input_frames2"] = remaining.pop(0)
            if "V" not in video_prompt_type:
                video_prompt_type += "V"
            if "+" not in video_prompt_type:
                video_prompt_type += "+"
        if remaining:
            raise ValueError(
                f"MiniMax H3 RefMod: {len(remaining)} video-kind RefMod(s) could not be placed -- "
                f"both native reference-video slots (Reference/Control Video 1 and 2) are already "
                f"used by the current generation settings or by other selected video-kind mods. "
                f"MiniMax H3 only supports 2 video reference slots in total; deselect a video-kind "
                f"RefMod, or free a live reference-video slot, to inject it.")
        kwargs["video_prompt_type"] = video_prompt_type

    _log(f"injecting {len(image_sentinels)} image-kind reference(s) + {len(video_sentinels)} "
         f"video-kind RefMod(s) (each in its own native reference-video slot, same as "
         f"'Use Two Reference Videos'), retention={retention:.2f} (~{total_tokens} tokens)")
    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _encode_ref_image(pipeline_self, video_cthw: torch.Tensor) -> torch.Tensor:
    """[C, 1, H, W] pixel tensor -> [1, 24, 1, H, W] VAE latent (cpu)."""
    return pipeline_self._encode_video(video_cthw)


def _encode_ref_video(pipeline_self, video_cthw: torch.Tensor) -> torch.Tensor:
    """[C, T, H, W] pixel tensor -> [1, 24, T', H, W] VAE latent (cpu).
    Frame count is snapped to the VAE's causal 4k+1 grid first."""
    t = video_cthw.shape[1]
    valid_t = core.snap_to_causal_grid(t)
    if valid_t != t:
        video_cthw = video_cthw[:, :valid_t]
    return pipeline_self._encode_video(video_cthw)


def _run_extract_job(pipeline_self, spec: dict, set_progress_status=None) -> None:
    if isinstance(spec, str):
        spec = json.loads(spec)

    def status(msg):
        _log(msg)
        if set_progress_status is not None:
            try:
                set_progress_status(msg)
            except Exception:
                pass

    name = storage._sanitize_name(spec.get("name") or "my_concept")
    mode = core.normalize_mode(spec.get("mode", "training"))
    concept_type = spec.get("concept_type", "generic")
    image_paths = [p for p in (spec.get("image_paths") or []) if p]
    video_paths = [p for p in (spec.get("video_path"), spec.get("video_path2")) if p]
    ref_resolution = int(spec.get("ref_resolution", 1024))
    pool_h = int(spec.get("pool_h", 16))
    pool_w = int(spec.get("pool_w", 16))
    latent_frames = int(spec.get("latent_frames", 16))
    identity = int(spec.get("identity", 500))
    multiplier = max(1, int(spec.get("multiplier", 1)))
    max_tokens = int(spec.get("max_tokens", 5120))
    description = str(spec.get("description", "") or "")
    save = bool(spec.get("save", True))
    remove_background_images_ref = int(spec.get("remove_background_images_ref", 0) or 0)

    if not image_paths and not video_paths:
        raise ValueError("RefMod extraction: no reference image(s) or video provided.")

    pool_warning = core.identity_training_pool_warning(concept_type, mode, pool_h, pool_w)
    if pool_warning:
        status(f"H3 RefMod: warning: {pool_warning}")

    status(f"H3 RefMod: loading {len(image_paths)} image(s)"
           + (f" + {len(video_paths)} video(s)" if video_paths else "") + f" (mode={mode})")

    rembg_session = None
    if remove_background_images_ref and image_paths:
        try:
            rembg_session = storage.new_rembg_session()
        except Exception as e:
            status(f"H3 RefMod: could not initialize background removal ({e!r}); "
                  f"continuing with backgrounds kept")
            remove_background_images_ref = 0

    sources = []  # list of (cthw_tensor, is_video)
    for i, p in enumerate(image_paths):
        from PIL import Image
        with Image.open(p) as img:
            img = img.copy()
        if remove_background_images_ref:
            status(f"H3 RefMod: removing background from image {i + 1}/{len(image_paths)}")
            img = storage.remove_background_from_image(img, session=rembg_session)
        sources.append((storage.pil_to_cthw(img), False))
    for vp in video_paths:
        video = storage.load_video_cthw(vp, max_frames=max(64, latent_frames * 8))
        # Take a CONTIGUOUS prefix matching the requested duration, for both modes -- not a
        # sparse sample spread across the whole clip. The old encode-mode behavior picked
        # `latent_frames` frames evenly spaced across the *entire* source video, then handed
        # them to the causally-compressing video VAE as if they were sequential -- the VAE has
        # no idea they were sparse, so it compresses them as a ~1-2s clip regardless of how
        # long a span they were actually pulled from, silently breaking the "duration to use"
        # slider's promise. Truncating up front instead keeps the number honest in both modes.
        target_px = (latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1
        if video.shape[1] > target_px:
            video = video[:, :target_px]
        sources.append((video, video.shape[1] > 1))

    canvas = None
    if mode == "encode" and len(sources) > 1:
        c, t0, h0, w0 = sources[0][0].shape
        scale = min(1.0, ref_resolution / min(h0, w0))
        canvas = (max(32, round(w0 * scale / 32) * 32), max(32, round(h0 * scale / 32) * 32))

    pool_grid = None
    if mode == "training":
        c, t0, h0, w0 = sources[0][0].shape
        pool_grid = core.aspect_grid(pool_h, pool_w, h0 / w0)
        if pool_grid != (pool_h, pool_w):
            status(f"pooled grid {pool_h}x{pool_w} -> {pool_grid[0]}x{pool_grid[1]} to match source aspect")
    gh, gw = pool_grid if pool_grid is not None else (pool_h, pool_w)

    frames = []
    n_img = n_vid = 0
    source_shapes = []
    for i, (src, is_video) in enumerate(sources):
        label = f"ref {i + 1}/{len(sources)} ({'video' if is_video else 'image'})"
        status(f"H3 RefMod: encoding {label}")
        if mode == "encode":
            src = storage.resize_cthw(src, ref_resolution, canvas)
        else:
            src = storage.resize_cthw(src, ref_resolution, None)
        src = storage.ensure_min_size(src)
        if is_video and src.shape[1] > 1:
            z = _encode_ref_video(pipeline_self, src)
        else:
            z = _encode_ref_image(pipeline_self, src[:, :1])
        if z.dim() != 5 or z.shape[1] != 24:
            raise ValueError(f"Expected a MiniMax H3 video-VAE latent [1,24,T,H,W], got {tuple(z.shape)}.")
        source_shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")

        if mode == "encode":
            pooled = z.to(torch.float16)
        else:
            pool_t = min(latent_frames, z.shape[2]) if is_video else 1
            pooled = core.pool_latent(z, pool_t, gh, gw).to(torch.float16)
            if identity > 0:
                status(f"H3 RefMod: refining identity for {label} ({identity} steps)")
                pooled = core.optimize_latent(pooled, z.float(), steps=identity, progress_every=100)
        frames.append(pooled)
        if is_video:
            n_vid += 1
        else:
            n_img += 1

    latent = torch.cat(frames, dim=2)
    if multiplier > 1:
        latent = latent.repeat(1, 1, multiplier, 1, 1)
    requested_t = latent.shape[2]
    if max_tokens > 0:
        latent, budget_messages = core.fit_token_budget(latent, max_tokens, name)
        for msg in budget_messages:
            status(f"H3 RefMod: {msg}")

    total_t = latent.shape[2]
    if total_t < requested_t:
        req_sec = ((requested_t - 1) * 4 + 1 if requested_t > 1 else 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE
        got_sec = ((total_t - 1) * 4 + 1 if total_t > 1 else 1) / FPS_ASSUMED_FOR_DURATION_ESTIMATE
        status(f"H3 RefMod: token budget ({max_tokens}) cut this mod short -- requested "
              f"~{req_sec:.1f}s worth of frames ({requested_t}), saved ~{got_sec:.1f}s "
              f"({total_t}). Raise 'Max tokens' (Advanced) or lower the ref resolution/pool "
              f"grid to keep more of the requested duration.")
    kind = "video" if n_vid > 0 else "image"  # NOT total_t > 1: several still images
                                              # stacked together (n_vid==0) still form
                                              # multiple independent *image* references,
                                              # not a multi-frame video, even though the
                                              # underlying latent has more than one frame.
    px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
    mod = core.H3RefMod(
        name=name, kind=kind, latent=latent, latent_h=latent.shape[3], latent_w=latent.shape[4],
        latent_t=total_t, mode=mode,
        source="stack" if len(frames) > 1 else ("video" if n_vid else "image"),
        source_shape=" +".join(source_shapes),
        pool=(f"full-res {px_w}x{px_h}px (short-edge cap {ref_resolution}px)" if mode == "encode"
              else f"{total_t}x{gh}x{gw}"),
        optimize_steps=identity if mode == "training" else 0,
        tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{multiplier} repeat"] if multiplier > 1 else [])
             + (["background removed"] if remove_background_images_ref else []),
        description=description, concept_type=concept_type,
    )

    if save:
        path = mod.save(storage.mod_path(name))
        status(f"H3 RefMod '{name}' saved: {mod.token_count} tokens, {kind}/{mode} -> {path}")
    else:
        status(f"H3 RefMod '{name}' extracted ({mod.token_count} tokens) but not saved (save=false)")
