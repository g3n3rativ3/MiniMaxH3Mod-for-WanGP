"""
MiniMax H3 RefMods -- Wan2GP plugin.

No-training "reference mods" for MiniMax H3 Ref2VA: compress an image/video
reference into a small .safetensors file once (Extract tab), then reuse it at
any strength in later generations without re-encoding it every time (Generate
tab), optionally blending several mods together. Ported from the ComfyUI
custom-node pack ComfyUI-MiniMaxH3Mod (MIT, (c) 2026 Luisa/luisacaotica) --
see README.md for how the mechanism works and its current limitations.

Design note on why this plugin has its own "Generate" section instead of
hooking the main Media Generator form: the per-model "Custom Settings" fields
Wan2GP auto-builds from a model's definition (the channel this plugin uses to
carry a RefMod selection all the way to the pipeline) are not given a stable
elem_id, so a plugin cannot bind its own rich widgets to them. Submitting a
self-contained task through the API session (the same mechanism the bundled
Sample Plugin demonstrates) sidesteps that limitation entirely and is the
supported, documented way for a plugin to drive a full generation.

The Generate panel below covers the fields most people tune day to day
(prompt, resolution, frame count, steps, flow shift, sampler, reference-image
budget, LoRAs, step-skipping accelerators, sliding window, and the sol-attn
sparsity dial) plus the RefMods themselves. Anything not exposed here still
gets a valid value: "Sync from the main form" copies every setting from
Wan2GP's own Media Generator tab (for the same model) as the starting point,
and `api_session.merge_settings_with_defaults(...)` fills in the rest from
that model's own factory defaults before submission -- so no field is ever
missing, even ones this panel doesn't have a dedicated widget for.
"""

from __future__ import annotations

import json

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import core, storage
from .patches import (SETTING_EXTRACT, SETTING_GENERATE, STASH_KEY, install_patches,
                      install_get_model_settings_patch, install_prepare_inputs_dict_patch,
                      is_minimax_h3_ref2va)

PlugIn_Name = "MiniMax H3 RefMods"
PlugIn_Id = "H3RefMods"

IMAGE_ROWS = 9  # matches MiniMax H3 Ref2VA's own native cap on image-kind references
VIDEO_ROWS = 2  # matches MiniMax H3 Ref2VA's own native cap on video-kind references
AUDIO_ROWS = 2  # matches MiniMax H3 Ref2VA's own native cap on audio-kind references
NONE_CHOICE = "(none)"

# Mirrors models/minimax_h3/minimax_h3_handler.py -- kept as a local constant
# so this UI doesn't need a live import of Wan2GP internals just to draw a
# dropdown. If a future Wan2GP version changes these, only this dropdown's
# labels/values would need updating, nothing else in the plugin depends on it.
FIRST_BLOCK_CACHE_STRENGTHS = [
    ("Low (0.06)", 0.06),
    ("Balanced (0.08, upstream default)", 0.08),
    ("High (0.10)", 0.10),
    ("Very High (0.12)", 0.12),
    ("Maximum (0.14)", 0.14),
]
STEPS_SKIPPING_CHOICES = [
    ("None", ""),
    ("Spectrum Feature Forecasting", "spectrum"),
    ("First Block Cache", "first_block"),
]
SAMPLE_SOLVER_CHOICES = [
    ("Euler", "euler"),
    ("RES Multistep", "res_multistep"),
    ("Ralston 2S (~2x slower)", "ralston_2s"),
]


def _diagnose(api_session, model_type, patch_error):
    """Confirm the plugin's monkeypatches are actually wired up for the
    selected model, *before* running an extraction/generation -- surfaces the
    exact failure mode that made the original bug so hard to notice (nothing
    ever raised an error; generate() just quietly ran a normal render)."""
    if not model_type:
        return "⚠️ Pick a model above first."
    lines = []
    if patch_error:
        lines.append(f"❌ Pipeline patch failed at plugin startup: {patch_error}")
    else:
        lines.append("✅ Pipeline patched (extraction / injection hooks are active).")
    try:
        model_def = api_session.get_model_def(model_type) or {}
        declared_ids = {s.get("id") for s in (model_def.get("custom_settings") or []) if isinstance(s, dict)}
        missing = [sid for sid in (SETTING_GENERATE, SETTING_EXTRACT) if sid not in declared_ids]
        if missing:
            lines.append(f"❌ This model does NOT declare {missing} under custom_settings -- RefMods will "
                         f"silently no-op for it (a real generation will run instead of an extraction, or "
                         f"without the mods applied). Check the terminal log for '[H3RefMod]' lines at "
                         f"Wan2GP startup -- the model-definition patch did not take effect for "
                         f"'{model_type}'.")
        else:
            lines.append(f"✅ '{model_type}' declares both custom_settings ids -- RefMod payloads will "
                         f"survive task submission (both this plugin's own Generate tab and, "
                         f"combined with the check below, the inline panel).")
    except Exception as e:
        lines.append(f"⚠️ Could not read the model definition for '{model_type}': {e!r}")
    try:
        from . import patches
        if getattr(patches.install_prepare_inputs_dict_patch, patches._PREPARE_INPUTS_PATCH_MARKER, False):
            lines.append("✅ Inline panel hook active (prepare_inputs_dict patched) -- RefMods selected "
                         "in the 'MiniMax H3 RefMods (inline)' accordion on the Media Generator page will "
                         "apply to that page's own Generate button.")
        else:
            lines.append("❌ Inline panel hook NOT active -- the inline accordion on the Media Generator "
                         "page will be visible but selections made there will have no effect. Use this "
                         "plugin's own 'Generate' tab instead, or check the terminal log for a "
                         "'[H3RefMod] prepare_inputs_dict' line explaining why.")
    except Exception as e:
        lines.append(f"⚠️ Could not check the inline panel hook status: {e!r}")
    return "\n".join(lines)


def _mod_choices(kind=None):
    """[(none), name, name, ...], optionally restricted to mods of a given
    "image"/"video"/"audio" kind so a slot can only ever offer mods that
    fit it."""
    if kind in ("image", "video", "audio"):
        names = storage.list_refmods_by_kind(kind)
    else:
        names = storage.list_refmods()
    return [NONE_CHOICE] + names


def _refresh_mod_dropdown_updates():
    """gr.update(...) for every mod-picker dropdown built by
    _build_mod_picker_rows, in the same image-then-video-then-audio order."""
    return ([gr.update(choices=_mod_choices("image")) for _ in range(IMAGE_ROWS)]
           + [gr.update(choices=_mod_choices("video")) for _ in range(VIDEO_ROWS)]
           + [gr.update(choices=_mod_choices("audio")) for _ in range(AUDIO_ROWS)])


FPS_ASSUMED_FOR_DURATION_ESTIMATE = 24  # MiniMax H3's own default fps -- only used to turn a
                                        # video-kind mod's latent frame count into an estimated
                                        # seconds figure for the counter below; the real cap
                                        # generate() enforces is duration-based (<=15s), not a
                                        # frame count, and uses whatever fps the render actually
                                        # runs at.

MAX_LATENT_FRAMES = 90  # the highest "latent frames" value that still stays under MiniMax H3's
                        # native 15s reference cap once the video VAE's causal 4:1 temporal
                        # compression is accounted for -- see latent_frames_to_seconds() below;
                        # 91 already estimates to just over 15s.


def latent_frames_to_seconds(latent_frames, fps: int = FPS_ASSUMED_FOR_DURATION_ESTIMATE) -> float:
    """Same causal-VAE math the live reference-budget counter uses (see
    _format_ref_counter): approximately how many real seconds of source
    video a given "latent frames" extraction setting corresponds to."""
    latent_frames = max(1, int(latent_frames))
    t_px = (latent_frames - 1) * 4 + 1 if latent_frames > 1 else 1
    return round(t_px / fps, 1)


def seconds_to_latent_frames(seconds, fps: int = FPS_ASSUMED_FOR_DURATION_ESTIMATE) -> int:
    """Inverse of latent_frames_to_seconds() -- what "latent frames" value
    to actually extract with so the result is close to the requested number
    of seconds. Round-trips exactly for every value latent_frames_to_seconds
    itself can produce."""
    seconds = max(0.0, float(seconds))
    t_px = seconds * fps
    if t_px <= 1:
        return 1
    return max(1, round((t_px - 1) / 4) + 1)


AUDIO_LATENTS_PER_SECOND = 40  # MiniMax H3's own audio VAE: encoder downsamples by 800x at
                               # 32kHz = 40 latents/s exactly (models/minimax_h3/components/
                               # audio_autoencoder.py's own docstring) -- unlike the video
                               # estimate below, this is an exact, fps-independent rate, not
                               # an approximation.


def _format_ref_counter(row_pairs):
    """A live 'how close to MiniMax H3 Ref2VA's own native reference caps am
    I' readout, from the current (mod_name, strength) values of every
    picker row (image rows, then video rows, then audio rows, any order
    internally). Mirrors exactly what _inject_refmods will actually send at
    generation time: rows with no mod picked or strength<=0 are skipped,
    and -- critically -- an image-kind mod counts once *per frame it
    contains* (see patches.py's _inject_refmods: a multi-image stack gets
    split into one reference per frame), not once per mod. Video- and
    audio-kind mods, by contrast, each go into their own native reference
    slot no matter how many are picked, so what actually matters for them
    is total duration (each against its own, separate 15-second budget --
    video and audio don't share one)."""
    n_images = 0
    video_latent_frames = 0
    audio_seconds_total = 0.0
    for name, strength in row_pairs:
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            continue
        if not name or name == NONE_CHOICE or strength <= 0:
            continue
        try:
            meta = storage.read_refmod_meta(storage.mod_path(name))
        except Exception:
            meta = None
        if meta is None:
            continue
        kind = meta.get("kind", "image")
        latent_t = max(1, int(meta.get("latent_t", 1)))
        if kind == "image":
            n_images += latent_t
        elif kind == "video":
            t_px = (latent_t - 1) * 4 + 1 if latent_t > 1 else 1  # undo the causal 4:1 compression
            video_latent_frames += t_px
        else:  # "audio"
            audio_seconds_total += latent_t / AUDIO_LATENTS_PER_SECOND
    video_seconds = video_latent_frames / FPS_ASSUMED_FOR_DURATION_ESTIMATE
    img_mark = "⚠️" if n_images > 9 else "▫️" if n_images == 0 else "✅"
    vid_mark = "⚠️" if video_seconds > 15 else "▫️" if video_seconds == 0 else "✅"
    aud_mark = "⚠️" if audio_seconds_total > 15 else "▫️" if audio_seconds_total == 0 else "✅"
    warn = ""
    if n_images > 9:
        warn += " -- **too many images, generation will fail.** A multi-image mod counts once per image it contains."
    if video_seconds > 15:
        warn += " -- **too much video, generation will fail.**"
    if audio_seconds_total > 15:
        warn += " -- **too much audio, generation will fail.**"
    return (f"{img_mark} **Images: {n_images} / 9**&nbsp;&nbsp;&nbsp;"
           f"{vid_mark} **Video: ~{video_seconds:.1f}s / 15s** (estimated at {FPS_ASSUMED_FOR_DURATION_ESTIMATE}fps)&nbsp;&nbsp;&nbsp;"
           f"{aud_mark} **Audio: {audio_seconds_total:.1f}s / 15s**"
           f"{warn}")


def _model_choices(api_session):
    try:
        records = api_session.list_model_defs(
            base_model_type=["minimax_h3_ref2va", "minimax_h3_ref2va_pruned"])
    except Exception as e:
        print(f"[H3RefMod] could not list MiniMax H3 Ref2VA models: {e!r}")
        records = []
    return [(r.get("name") or r.get("model_type"), r.get("model_type")) for r in records]


DEFAULT_MODEL_TYPE = "minimax_h3_ref2va_pruned"  # "MiniMax H3 Ref2VA Pruned 20B"


def _default_model_choice(model_choices):
    """Preselect MiniMax H3 Ref2VA Pruned 20B at startup -- the lighter,
    faster variant most people running this plugin day to day will have
    installed. Falls back to whatever's first if that exact model, or any
    other model whose name mentions "pruned", isn't available."""
    for _, model_type in model_choices:
        if model_type == DEFAULT_MODEL_TYPE:
            return model_type
    for display_name, model_type in model_choices:
        if "pruned" in str(display_name).lower():
            return model_type
    return model_choices[0][1] if model_choices else None


def _library_rows():
    return [[i["name"], i["kind"], i["mode"], i["tokens"], i["size_mb"], i["concept_type"], i["description"]]
            for i in storage.list_refmods_info()]


def _lora_choices(api_session, model_type):
    if not model_type:
        return []
    try:
        info = api_session.list_loras(model_type)
    except Exception as e:
        print(f"[H3RefMod] could not list LoRAs for {model_type}: {e!r}")
        return []
    return list(info.get("loras") or [])


class MiniMaxH3RefModsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.26.0"
        self.description = ("No-training reference mods for MiniMax H3: compress a reference "
                            "into a small file once, reuse it at any strength without "
                            "re-encoding it every generation.")
        self._patch_error = install_patches()

    def setup_ui(self):
        self.request_component("state")
        self.request_component("model_choice_target")
        self.request_component("wangp_model_choice_target")  # legacy/alternate name, harmless if absent
        self.request_global("get_current_model_settings")
        self.request_global("refresh_model_defs")
        self.request_global("prepare_inputs_dict")
        self.request_global("get_state_model_type")
        self.request_global("get_model_settings")
        self.request_global("get_base_model_type")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)
        self.insert_after(target_component_id="loras_multipliers",
                          new_component_constructor=self._build_inline_refmods_section)

    def post_ui_setup(self, components):
        """Wan2GP builds its whole model catalog (``models_def``, what
        ``get_model_def()`` reads from) once, at import time, well before any
        plugin is loaded -- so the ``family_handler.query_model_def`` patch
        installed in ``__init__`` (see patches.py) has no effect on entries
        already cached by then, even though the patch itself is correctly in
        place. ``self.refresh_model_defs`` (a global, only available once
        Wan2GP finishes injecting plugin globals -- i.e. here, not in
        ``__init__``) is Wan2GP's own supported way to rebuild that catalog
        on demand; calling it once now forces every MiniMax H3 model
        definition to be recomputed through our now-patched function, so the
        two RefMod custom_settings actually end up in the cache the rest of
        Wan2GP reads from."""
        if getattr(self, "_post_ui_setup_done", False):
            return {}  # Wan2GP appears to call post_ui_setup more than once at startup;
                      # everything below is safe to repeat but noisy/wasteful, so skip it.
        self._post_ui_setup_done = True

        refresh = getattr(self, "refresh_model_defs", None)
        if callable(refresh):
            try:
                refresh()
                print("[H3RefMod] refreshed Wan2GP's model catalog so the RefMod custom_settings "
                     "declaration takes effect on cached MiniMax H3 model definitions")
            except Exception as e:
                print(f"[H3RefMod] refresh_model_defs() failed ({e!r}); MiniMax H3 model "
                     "definitions may still be missing the RefMod custom_settings -- use "
                     "'Check setup for this model' in the plugin tab to confirm.")
        else:
            print("[H3RefMod] refresh_model_defs was not exposed by Wan2GP's plugin globals; "
                 "RefMods will likely not take effect until Wan2GP is restarted after this "
                 "plugin was enabled. Use 'Check setup for this model' in the plugin tab to confirm.")

        get_base_mt = getattr(self, "get_base_model_type", None)
        if not callable(get_base_mt):
            print("[H3RefMod] get_base_model_type was not exposed by Wan2GP's plugin globals; "
                 "a MiniMax H3 Ref2VA finetune whose own model_type name doesn't start with "
                 "'minimax_h3_ref2va' (e.g. a custom-named checkpoint) may not be recognized -- "
                 "the inline panel would stay hidden for it, and RefMods wouldn't apply even if "
                 "selected through the plugin's own 'Generate' tab.")
            get_base_mt = None

        orig_prepare = getattr(self, "prepare_inputs_dict", None)
        get_state_mt = getattr(self, "get_state_model_type", None)
        if callable(orig_prepare) and callable(get_state_mt):
            err = install_prepare_inputs_dict_patch(orig_prepare, get_state_mt, self.set_global, get_base_mt)
            if err:
                print(f"[H3RefMod] {err}")
        else:
            print("[H3RefMod] prepare_inputs_dict / get_state_model_type were not exposed by "
                 "Wan2GP's plugin globals; the inline RefMods panel on the Media Generator page "
                 "will be visible but will not affect generations from that page's own Generate "
                 "button. Use the plugin's own 'Generate' tab instead.")

        orig_get_model_settings = getattr(self, "get_model_settings", None)
        if callable(orig_get_model_settings):
            err2 = install_get_model_settings_patch(orig_get_model_settings, self.set_global, get_base_mt)
            if err2:
                print(f"[H3RefMod] {err2}")
        else:
            print("[H3RefMod] get_model_settings was not exposed by Wan2GP's plugin globals; "
                 "a RefMod change right before clicking Generate (with no other field touched "
                 "in between) may not always be picked up from the inline panel -- change any "
                 "other field (e.g. click into the prompt box) once after picking mods as a "
                 "workaround, or use the plugin's own 'Generate' tab instead.")
        return {}

    # ── Inline panel injected onto the Media Generator page ───────────────

    def _build_inline_refmods_section(self):
        # Wan2GP hands plugins their requested components out of generate_media_tab's
        # own locals(), so the key is the Python *variable* name ("model_choice_target"),
        # not the elem_id ("wangp_model_choice_target"). Try both so this works across
        # Wan2GP versions regardless of which name a given build exposes.
        target = getattr(self, "model_choice_target", None) or getattr(self, "wangp_model_choice_target", None)
        # Default to visible (matches the previous, always-shown behavior) rather than
        # hidden: there's no confirmed signal that fires on the very first page load
        # (only on an actual model switch), so defaulting to hidden could strand the
        # panel out of sight for anyone who already has a MiniMax H3 Ref2VA model
        # selected by default and hasn't touched the model dropdown yet. The .change()
        # handler below still correctly hides it the moment any model switch happens.
        with gr.Accordion("MiniMax H3 RefMods (inline)", open=False) as accordion:
            gr.Markdown(
                "Applies only when a **MiniMax H3 Ref2VA** model is selected above -- every other "
                "field on this page (resolution, frame count, steps, attention mode, memory "
                "profile, output filename, etc.) is untouched and works exactly as normal. "
                "Extract new RefMods from the **MiniMax H3 RefMods** tab. Selecting a mod here "
                "applies immediately to generations started from *this page's* own Generate "
                "button -- no separate submission needed.")
            mod_rows = self._build_mod_picker_rows()
            status = gr.Markdown("*No RefMods selected.*")

        widgets = [c for row in mod_rows for c in row]

        def apply_selection(state, *vals):
            rows = []
            n_rows = IMAGE_ROWS + VIDEO_ROWS + AUDIO_ROWS
            for i in range(n_rows):
                mdd, strength = vals[i * 2], vals[i * 2 + 1]
                if mdd and mdd != NONE_CHOICE and strength > 0:
                    rows.append({"mod": mdd, "strength": float(strength)})
            payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None}
            if not isinstance(state, dict):
                return state, "⚠️ Could not access session state -- try reloading the page."
            if rows:
                state[STASH_KEY] = json.dumps(payload)
                msg = f"✅ {len(rows)} RefMod(s) armed for the next MiniMax H3 Ref2VA generation from this page."
            else:
                state.pop(STASH_KEY, None)
                msg = "*No RefMods selected.*"
            return state, msg

        for w in widgets:
            w.change(fn=apply_selection, inputs=[self.state] + widgets, outputs=[self.state, status], queue=False)

        if target is not None:
            get_base_mt = getattr(self, "get_base_model_type", None)

            def update_visibility(target_value):
                model_type = str(target_value or "").split("|", 1)[0].strip()
                return gr.update(visible=is_minimax_h3_ref2va(model_type, get_base_mt))

            target.change(fn=update_visibility, inputs=[target], outputs=[accordion], queue=False)
        else:
            print("[H3RefMod] neither 'model_choice_target' nor 'wangp_model_choice_target' was "
                 "exposed by Wan2GP -- the inline panel will stay visible for every model instead "
                 "of only MiniMax H3 Ref2VA ones (it still has no effect on other models, this "
                 "only affects whether it's shown).")

        return accordion

    def _submit(self, api_session, model_type, overrides, callbacks):
        """Build a full, validated settings dict for ``model_type`` -- starting
        from that model's own factory defaults, then applying ``overrides`` --
        and submit it. Any field this plugin doesn't have a widget for still
        gets a sane, model-correct value this way."""
        payload = dict(overrides)
        payload["model_type"] = model_type  # always wins, even if overrides (e.g. a synced
                                            # main-form snapshot) carried a different one
        merged = api_session.merge_settings_with_defaults(payload)
        merged["model_type"] = model_type
        job = api_session.submit_task(merged, callbacks=callbacks)
        return job.result()

    # ── Extract ─────────────────────────────────────────────────────────

    def _build_mod_picker_rows(self):
        """A "Refresh mod list" button + a live reference-budget counter,
        followed by IMAGE_ROWS image-kind + VIDEO_ROWS video-kind +
        AUDIO_ROWS audio-kind mod picker rows (dropdown restricted to that
        kind, plus strength), matching MiniMax H3 Ref2VA's own native
        reference caps. The refresh button and counter live here (above the
        rows, for visibility) and are fully wired before returning. Returns
        the flat list of (dropdown, strength) pairs, image rows first, then
        video rows, then audio rows -- callers must keep that same order
        when reading values back (_refresh_mod_dropdown_updates() above
        does too)."""
        refresh_btn = gr.Button("🔄 Refresh mod list", size="sm")
        counter = gr.Markdown(_format_ref_counter([]))
        mod_rows = []
        gr.Markdown(f"**Image RefMods** (up to {IMAGE_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(IMAGE_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("image"), value=NONE_CHOICE,
                                  label=f"Image Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))
        gr.Markdown(f"**Video RefMods** (up to {VIDEO_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(VIDEO_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("video"), value=NONE_CHOICE,
                                  label=f"Video Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))
        gr.Markdown(f"**Audio RefMods** (up to {AUDIO_ROWS}; leave a row on `{NONE_CHOICE}` to skip it)")
        for i in range(AUDIO_ROWS):
            with gr.Row():
                mdd = gr.Dropdown(choices=_mod_choices("audio"), value=NONE_CHOICE,
                                  label=f"Audio Mod {i + 1}", scale=3)
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Strength", scale=2)
                mod_rows.append((mdd, strength))
        refresh_btn.click(fn=_refresh_mod_dropdown_updates, outputs=[r[0] for r in mod_rows], queue=False)

        def update_counter(*vals):
            n = len(mod_rows)
            pairs = [(vals[i * 2], vals[i * 2 + 1]) for i in range(n)]
            return _format_ref_counter(pairs)

        row_widgets = [c for row in mod_rows for c in row]
        for w in row_widgets:
            w.change(fn=update_counter, inputs=row_widgets, outputs=[counter], queue=False)

        return mod_rows

    def _build_extract_section(self, api_session, model_dd):
        gr.Markdown("### Extract a RefMod\n"
                    "Turn one or more reference images (and/or one reference video) into a small "
                    "saved file. This briefly runs a real generation task on the model selected "
                    "above so it can reuse its already-loaded VAE -- you'll see the usual progress "
                    "bar for a few seconds, then **no video is produced on purpose**: the mod file "
                    "is what was created. Check the status line below for confirmation.")
        with gr.Row():
            name = gr.Textbox(label="Mod name", value="my_concept")
            mode = gr.Radio(label="Mode", choices=["training", "encode"], value="training",
                            info="encode = full fidelity (identical to a live reference), best for a "
                                 "precise face/identity. training = approximate (pooling discards "
                                 "fine detail), best for a general concept/style/pose where "
                                 "approximation is fine.")
        with gr.Row():
            ref_images = gr.Files(label="Reference image(s)", file_types=["image"], file_count="multiple")
            ref_video = gr.Video(label="Reference video 1 (optional)")
            ref_video2 = gr.Video(label="Reference video 2 (optional)")
        gr.Markdown("*MiniMax H3 Ref2VA natively supports up to two reference videos "
                   "('Use Two Reference Videos') -- both are encoded and combined into this one mod.*")
        ref_audio = gr.Audio(label="Reference audio (optional)", type="filepath")
        gr.Markdown("*Audio can't be combined with image/video sources in the same mod (their "
                   "encoded shapes are structurally different) -- providing audio here along "
                   "with images/video is refused with a clear error rather than mixed. Always "
                   "extracted at full fidelity; 'Mode' above doesn't apply to audio.*")
        audio_duration_warning = gr.Markdown("")
        remove_background_images_ref = gr.Dropdown(
            choices=[("Keep Backgrounds behind all Reference Images", 0),
                    ("Remove Background behind People / Objects", 1)],
            value=0,
            label="Automatic Removal of Background behind People or Objects in Reference Images",
            info="Same background removal Wan2GP's own Media Generator uses for reference images "
                 "-- only applies to the image(s) above, not to reference videos.")
        with gr.Accordion("Advanced", open=True):
            with gr.Row():
                ref_resolution = gr.Slider(256, 2048, value=1024, step=64, label="Ref resolution (short edge, px)",
                                           info="Size before encoding. Smaller = faster, less detail.")
                pool_h = gr.Slider(2, 64, value=16, step=2, label="Pool grid height (training mode)",
                                   info="Training mode. Bigger grid = more detail, more tokens.")
                pool_w = gr.Slider(2, 64, value=16, step=2, label="Pool grid width (training mode)",
                                   info="Training mode. Same as height, other axis.")
            pool_warning = gr.Markdown("")
            with gr.Row():
                latent_frames = gr.Slider(0.1, latent_frames_to_seconds(MAX_LATENT_FRAMES),
                                          value=latent_frames_to_seconds(16), step=0.1,
                                          label="Reference duration to use (seconds) -- video AND audio",
                                          info="Approximate real seconds kept from the source, for a "
                                               "video-kind ref **or** an audio-kind ref (whichever you're "
                                               "extracting) -- always double-check this before extracting "
                                               "audio, since it defaults to a short video-sized value and "
                                               "audio has no separate duration control of its own. "
                                               "MiniMax H3's native 15s reference cap is a **shared** "
                                               "budget within its own kind -- **if a generation combines "
                                               "two video-kind (or two audio-kind) mods, their durations "
                                               "add together, so one mod near 14.9s leaves no room for a "
                                               "second one of the same kind alongside it.**")
                identity = gr.Slider(0, 2000, value=500, step=50, label="Identity refinement steps (training mode)",
                                     info="Training mode. Higher = truer to original.")
                multiplier = gr.Slider(1, 10, value=1, step=1, label="Repeat multiplier",
                                       info="Repeats the mod. Higher = stronger effect, bigger file.")
                max_tokens = gr.Slider(0, 65536, value=65536, step=512, label="Max tokens (0 = no cap)",
                                       info="Max tokens allowed. 0 = unlimited.")
            description = gr.Textbox(
                label="keyword - description", lines=2,
                info="Text note only. Model never reads it.")
        with gr.Accordion("Misc.", open=False):
            concept_type = gr.Dropdown(
                label="Concept type", choices=list(core.CONCEPT_TYPES), value="generic",
                info="Label only. No effect on generation.")
            save = gr.Checkbox(label="Save to disk", value=True,
                               info="Off = test run, nothing saved.")
        extract_btn = gr.Button("Extract & Save RefMod", variant="primary")
        extract_status = gr.Textbox(label="Status", interactive=False, lines=3)

        def update_pool_warning(concept_type, mode, pool_h, pool_w):
            w = core.identity_training_pool_warning(concept_type, mode, int(pool_h), int(pool_w))
            return f"⚠️ {w}" if w else ""

        for w in (concept_type, mode, pool_h, pool_w):
            w.change(fn=update_pool_warning, inputs=[concept_type, mode, pool_h, pool_w],
                    outputs=[pool_warning], queue=False)

        def update_audio_duration_warning(audio_path, current_seconds):
            if not audio_path:
                return ""
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                real_seconds = info.frames / info.samplerate
            except Exception:
                return ""
            if real_seconds > float(current_seconds) + 0.15:
                return (f"⚠️ This audio file is ~{real_seconds:.1f}s long, but 'Reference duration "
                       f"to use' above is only set to {float(current_seconds):.1f}s -- only the "
                       f"first {float(current_seconds):.1f}s will be kept. Raise it (up to "
                       f"{latent_frames_to_seconds(MAX_LATENT_FRAMES)}s) to use more of this "
                       f"recording.")
            return ""

        for w in (ref_audio, latent_frames):
            w.change(fn=update_audio_duration_warning, inputs=[ref_audio, latent_frames],
                    outputs=[audio_duration_warning], queue=False)

        def do_extract(model_type, name, mode, concept_type, ref_images, ref_video, ref_video2, ref_audio,
                       remove_background_images_ref, ref_resolution, pool_h, pool_w, latent_frames,
                       identity, multiplier, max_tokens, description, save):
            if not model_type:
                return "Pick a MiniMax H3 Ref2VA model above first."
            image_paths = [f.name if hasattr(f, "name") else f for f in (ref_images or [])]
            if not image_paths and not ref_video and not ref_video2 and not ref_audio:
                return "Add at least one reference image, video, or audio file."
            if ref_audio and (image_paths or ref_video or ref_video2):
                return ("Audio can't be combined with image/video sources in the same mod -- "
                       "clear the image/video fields to extract an audio-only mod, or clear "
                       "the audio field to extract an image/video mod.")
            spec = {
                "name": name, "mode": mode, "concept_type": concept_type,
                "image_paths": image_paths, "video_path": ref_video, "video_path2": ref_video2,
                "audio_path": ref_audio,
                "remove_background_images_ref": int(remove_background_images_ref or 0),
                "ref_resolution": int(ref_resolution), "pool_h": int(pool_h), "pool_w": int(pool_w),
                "latent_frames": seconds_to_latent_frames(latent_frames), "identity": int(identity),
                "multiplier": int(multiplier), "max_tokens": int(max_tokens),
                "description": description or "", "save": bool(save),
            }
            log = {"lines": []}

            class ExtractCallbacks:
                def on_status(self, status):
                    if status:
                        log["lines"].append(str(status))

                def on_progress(self, update):
                    pass

            try:
                self._submit(api_session, model_type,
                            {"video_length": 107, "custom_settings": {SETTING_EXTRACT: json.dumps(spec)}},
                            ExtractCallbacks())
            except Exception as e:
                return f"Extraction task failed to run: {e!r}"
            tail = "\n".join(log["lines"][-6:])
            ok = ("Done -- check the Library tab (Refresh) to see the saved mod." if any(
                    "saved" in l.lower() for l in log["lines"]) else
                  "Task finished, but no confirmation line was captured -- check the terminal log.")
            return (tail + ("\n" if tail else "") + ok) if tail else ok

        extract_btn.click(
            fn=do_extract,
            inputs=[model_dd, name, mode, concept_type, ref_images, ref_video, ref_video2, ref_audio,
                   remove_background_images_ref, ref_resolution, pool_h, pool_w, latent_frames,
                   identity, multiplier, max_tokens, description, save],
            outputs=[extract_status],
            queue=False,
        )

    # ── Library ─────────────────────────────────────────────────────────

    def _build_library_section(self):
        gr.Markdown("### Saved RefMods\n"
                    "Mods live in `loras/refmods_plugin/minimax_h3/`. "
                    "Mods produced by the ComfyUI-MiniMaxH3Mod pack use the same file format and "
                    "can be dropped into that folder directly.")
        table = gr.Dataframe(
            headers=["name", "kind", "mode", "tokens", "size (MB)", "concept type", "description"],
            value=_library_rows(), interactive=False, wrap=True)
        with gr.Row():
            refresh_btn = gr.Button("Refresh")
            delete_name = gr.Textbox(label="Mod name to delete", scale=2)
            delete_btn = gr.Button("Delete", variant="stop")
        delete_status = gr.Textbox(label="", interactive=False, show_label=False)
        gr.Markdown(
            "Mods extracted purely from several still images (no video source) used to be "
            "wrongly saved as `video` kind if more than one image was stacked together -- the "
            "button below rescans every saved mod's own extraction record and corrects it in "
            "place (the latent data itself is untouched, only the `kind` label). Safe to run "
            "anytime, including on mods that are already correct.")
        with gr.Row():
            fix_btn = gr.Button("Fix classification (image vs video)")
            fix_status = gr.Markdown("")

        gr.Markdown(
            "**Rename or edit a mod's description.** Pick a mod, click Load, edit either field, "
            "then Save -- the latent data itself is never touched, only the name/description. "
            "Both fields are plain editable text boxes, so you can also just click into either one, "
            "select the text (e.g. double/triple-click, or Ctrl+A), and copy it (Ctrl+C) to reuse "
            "elsewhere -- no separate copy button needed.")
        with gr.Row():
            edit_name_dd = gr.Dropdown(label="Mod to edit", choices=storage.list_refmods(), scale=2)
            edit_load_btn = gr.Button("Load")
        with gr.Row():
            edit_name_field = gr.Textbox(label="Name")
            edit_description_field = gr.Textbox(label="Description", lines=2)
        edit_save_btn = gr.Button("Save changes", variant="primary")
        edit_status = gr.Textbox(label="", interactive=False, show_label=False)

        # ── wiring (all components above already exist) ──
        refresh_btn.click(fn=lambda: (_library_rows(), gr.update(choices=storage.list_refmods())),
                          outputs=[table, edit_name_dd], queue=False)

        def do_delete(name):
            if not name:
                return _library_rows(), gr.update(choices=storage.list_refmods()), "Enter a mod name first."
            ok = storage.delete_refmod(name)
            return (_library_rows(), gr.update(choices=storage.list_refmods()),
                   (f"Deleted '{name}'." if ok else f"No mod named '{name}' found."))

        delete_btn.click(fn=do_delete, inputs=[delete_name], outputs=[table, edit_name_dd, delete_status], queue=False)

        def do_fix():
            fixed, checked = storage.reclassify_all_mods()
            msg = (f"Checked {checked} mod(s), fixed {fixed}." if fixed else
                  f"Checked {checked} mod(s), all already correctly classified.")
            return _library_rows(), gr.update(choices=storage.list_refmods()), msg

        fix_btn.click(fn=do_fix, outputs=[table, edit_name_dd, fix_status], queue=False)

        def do_load_for_edit(name):
            if not name:
                return "", "", "Pick a mod first."
            meta = storage.read_refmod_meta(storage.mod_path(name))
            if meta is None:
                return "", "", f"No mod named '{name}' found."
            return meta.get("name", name), meta.get("description", "") or "", ""

        edit_load_btn.click(fn=do_load_for_edit, inputs=[edit_name_dd],
                            outputs=[edit_name_field, edit_description_field, edit_status], queue=False)

        def do_save_edit(old_name, new_name, new_description):
            if not old_name:
                return _library_rows(), gr.update(), "Pick a mod first (use Load)."
            try:
                final_name = storage.rename_and_update_mod(old_name, new_name=new_name,
                                                            new_description=new_description)
            except Exception as e:
                return _library_rows(), gr.update(), f"Could not save: {e!r}"
            msg = f"Saved as '{final_name}'." if final_name != old_name else "Saved."
            return _library_rows(), gr.update(choices=storage.list_refmods(), value=final_name), msg

        edit_save_btn.click(fn=do_save_edit, inputs=[edit_name_dd, edit_name_field, edit_description_field],
                            outputs=[table, edit_name_dd, edit_status], queue=False)

        gr.Markdown(
            "**Build prompt hint.** The `keyword - description` field you set at extraction is "
            "never read by the model automatically -- there's no image-embedding channel on H3's "
            "Ref2VA path to hang an automatic clue off of, so a mod file's description only reaches "
            "the model if you type it into your prompt yourself. This builds that string for you "
            "from one or more saved mods (`concept_type: description; concept_type: description...`) "
            "so you can copy it into your prompt instead of retyping it by hand -- mirrors the "
            "ComfyUI pack's own \"prompt hint\" loader output.")
        with gr.Row():
            hint_names = gr.Textbox(label="Mod name(s), comma-separated, in the order you want them combined",
                                    scale=2)
            hint_btn = gr.Button("Build Prompt Hint")
        hint_output = gr.Textbox(label="Prompt hint -- copy this into your prompt", interactive=False, lines=2)

        def do_build_hint(names_str):
            names = [n.strip() for n in (names_str or "").split(",") if n.strip()]
            if not names:
                return "Enter at least one mod name."
            metas = []
            for n in names:
                meta = storage.read_refmod_meta(storage.mod_path(n))
                if meta is None:
                    return f"No mod named '{n}' found."
                metas.append(meta)
            hint = core.build_prompt_hint(metas)
            return hint or "(none of the selected mod(s) have a description set -- nothing to hint)"

        hint_btn.click(fn=do_build_hint, inputs=[hint_names], outputs=[hint_output], queue=False)
        return table

    # ── Generate ────────────────────────────────────────────────────────

    def _build_generate_section(self, api_session, model_dd):
        gr.Markdown(
            "# ⚠️ Prefer the **Media Generator** tab for actual generation\n"
            "**It has far more options (resolution categories, attention mode, memory profile, "
            "output filename, and everything else Wan2GP's main form offers) -- and your RefMods "
            "are available there too, under LoRAs → MiniMax H3 RefMods (inline).** This panel below "
            "is kept as a simpler fallback with only a subset of the fields.")
        gr.Markdown("### Generate with RefMods\n"
                    "Covers the fields you're most likely to tune here: prompt, resolution, frame "
                    "count, steps, sampler, reference-image budget, LoRAs, step-skipping "
                    "accelerators, sliding window, and sol-attn sparsity, plus up to "
                    f"{IMAGE_ROWS} image + {VIDEO_ROWS} video RefMods (MiniMax H3 Ref2VA's own "
                    "native caps). Anything else keeps whatever you last set on the main "
                    "**Media Generator** tab for this model (use *Sync from the main form* below), "
                    "or that model's own factory defaults otherwise -- so no setting is ever left "
                    "unset, even ones without a widget here.")
        synced = gr.State({})
        with gr.Row():
            sync_btn = gr.Button("⟲ Sync from the main form (copies every current setting)")
            sync_status = gr.Markdown("*Not synced yet -- using this model's factory defaults.*")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=4, scale=3)
            with gr.Column(scale=1):
                resolution = gr.Textbox(label="Resolution (WxH)", value="1280x720")
                video_length = gr.Slider(107, 737, value=124, step=17, label="Number of frames")
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                repeat_generation = gr.Slider(1, 25, value=1, step=1, label="Videos per prompt")

        with gr.Row():
            num_inference_steps = gr.Slider(1, 100, value=20, step=1, label="Number of inference steps")
            flow_shift = gr.Slider(0.0, 25.0, value=12.0, step=0.1, label="Flow shift")
            sample_solver = gr.Dropdown(label="Sampler solver / scheduler", choices=SAMPLE_SOLVER_CHOICES,
                                        value="euler")
            image_refs_relative_size = gr.Slider(50, 400, value=100, step=1,
                                                 label="Reference image budget (% of output pixels)",
                                                 info="Higher = more reference detail kept, slower.")

        mod_rows = self._build_mod_picker_rows()

        with gr.Accordion("Advanced Mode", open=False):
            with gr.Tab("LoRAs"):
                lora_choices = gr.State([])
                activated_loras = gr.Dropdown(label="Activated LoRAs", choices=[], multiselect=True, value=[])
                loras_multipliers = gr.Textbox(
                    label="LoRAs multipliers (1.0 by default) separated by spaces or line breaks", value="")
                refresh_loras_btn = gr.Button("Refresh LoRA list")

            with gr.Tab("Steps Skipping"):
                skip_steps_cache_type = gr.Dropdown(label="Skip steps cache type", choices=STEPS_SKIPPING_CHOICES,
                                                    value="")
                skip_steps_multiplier = gr.Dropdown(label="First Block Cache threshold",
                                                    choices=FIRST_BLOCK_CACHE_STRENGTHS, value=0.08,
                                                    visible=False)
                skip_steps_start_step_perc = gr.Slider(0, 100, value=25, step=1,
                                                       label="Skip steps starting moment (% of generation)")
                skip_steps_cache_type.change(
                    fn=lambda t: gr.update(visible=t == "first_block"),
                    inputs=[skip_steps_cache_type], outputs=[skip_steps_multiplier], queue=False)

            with gr.Tab("Sliding Window"):
                gr.Markdown("Used automatically once **Number of frames** needs more than one window.")
                sliding_window_size = gr.Slider(124, 481, value=362, step=17, label="Sliding window size (frames)")
                sliding_window_overlap = gr.Slider(1, 120, value=18, step=17, label="Sliding window overlap (frames)")

            with gr.Tab("Attention"):
                override_attention = gr.Dropdown(
                    label="Override attention mode", value="",
                    choices=[("Auto (recommended)", ""), ("Sol-Attn (sparse)", "sol")],
                    allow_custom_value=True,
                    info="Leave on Auto unless you know a specific backend is installed.")
                attention_sparsity = gr.Slider(
                    0.0, 4.0, value=1.3, step=0.05, label="Sol-Attn Start Tau",
                    info="Only used when Sol-Attn is selected above. Higher = sparser/faster, "
                         "lower = denser/more faithful. End Tau is fixed at 0.8.")

        generate_btn = gr.Button("Generate", variant="primary")
        output_video = gr.Video(label="Output")
        gen_status = gr.Textbox(label="Status", interactive=False, lines=3)

        def do_sync(state, model_type):
            settings = dict(self.get_current_model_settings(state) or {})
            note = ("*Synced from the main form.*" if settings else
                   "*Main form has no settings for the current model yet -- using factory defaults.*")
            return (settings, note, settings.get("prompt", ""), settings.get("resolution", "1280x720"),
                   settings.get("video_length", 124), settings.get("seed", -1),
                   settings.get("repeat_generation", 1), settings.get("num_inference_steps", 20),
                   settings.get("flow_shift", 12.0), settings.get("sample_solver", "euler"),
                   settings.get("image_refs_relative_size", 100),
                   settings.get("activated_loras", []), settings.get("loras_multipliers", ""),
                   settings.get("skip_steps_cache_type", ""), settings.get("skip_steps_multiplier", 0.08),
                   settings.get("skip_steps_start_step_perc", 25),
                   settings.get("sliding_window_size", 362), settings.get("sliding_window_overlap", 18),
                   settings.get("override_attention", ""), settings.get("attention_sparsity", 1.3))

        def do_refresh_loras(model_type):
            choices = _lora_choices(api_session, model_type)
            return choices, gr.update(choices=choices)

        def do_generate(model_type, synced_settings, prompt, resolution, video_length, seed, repeat_generation,
                        num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                        activated_loras, loras_multipliers,
                        skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                        sliding_window_size, sliding_window_overlap,
                        override_attention, attention_sparsity, *row_values):
            if not model_type:
                raise gr.Error("Pick a MiniMax H3 Ref2VA model above first.")
            rows = []
            n_rows = IMAGE_ROWS + VIDEO_ROWS + AUDIO_ROWS
            for i in range(n_rows):
                mdd, strength = row_values[i * 2], row_values[i * 2 + 1]
                if mdd and mdd != NONE_CHOICE and strength > 0:
                    rows.append({"mod": mdd, "strength": float(strength)})
            state_payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None}

            overrides = dict(synced_settings or {})
            overrides.update({
                "prompt": prompt, "resolution": resolution, "video_length": int(video_length),
                "seed": int(seed), "repeat_generation": int(repeat_generation),
                "num_inference_steps": int(num_inference_steps), "flow_shift": float(flow_shift),
                "sample_solver": sample_solver, "image_refs_relative_size": int(image_refs_relative_size),
                "activated_loras": list(activated_loras or []), "loras_multipliers": loras_multipliers or "",
                "skip_steps_cache_type": skip_steps_cache_type, "skip_steps_multiplier": float(skip_steps_multiplier),
                "skip_steps_start_step_perc": int(skip_steps_start_step_perc),
                "sliding_window_size": int(sliding_window_size), "sliding_window_overlap": int(sliding_window_overlap),
                "override_attention": override_attention, "attention_sparsity": float(attention_sparsity),
                "custom_settings": {SETTING_GENERATE: json.dumps(state_payload)},
            })

            log = {"lines": []}

            class GenCallbacks:
                def on_status(self, status):
                    if status:
                        log["lines"].append(str(status))

                def on_progress(self, update):
                    pass

            try:
                result = self._submit(api_session, model_type, overrides, GenCallbacks())
            except Exception as e:
                return gr.update(), f"Generation task failed to run: {e!r}"
            tail = "\n".join(log["lines"][-6:])
            if result.success and result.generated_files:
                return result.generated_files[0], tail or "Done."
            if result.cancelled:
                return gr.update(), ((tail + "\nCancelled.") if tail else "Cancelled.")
            errors = list(result.errors or [])
            return gr.update(), (tail + "\n" if tail else "") + str(errors[0] if errors else "No output produced.")

        sync_btn.click(
            fn=do_sync, inputs=[self.state, model_dd],
            outputs=[synced, sync_status, prompt, resolution, video_length, seed, repeat_generation,
                    num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                    activated_loras, loras_multipliers,
                    skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                    sliding_window_size, sliding_window_overlap, override_attention, attention_sparsity],
            queue=False,
        )
        refresh_loras_btn.click(fn=do_refresh_loras, inputs=[model_dd], outputs=[lora_choices, activated_loras],
                               queue=False)
        flat_rows = [c for row in mod_rows for c in row]
        generate_btn.click(
            fn=do_generate,
            inputs=[model_dd, synced, prompt, resolution, video_length, seed, repeat_generation,
                   num_inference_steps, flow_shift, sample_solver, image_refs_relative_size,
                   activated_loras, loras_multipliers,
                   skip_steps_cache_type, skip_steps_multiplier, skip_steps_start_step_perc,
                   sliding_window_size, sliding_window_overlap, override_attention, attention_sparsity] + flat_rows,
            outputs=[output_video, gen_status],
            queue=False,
        )

    # ── Tab assembly ────────────────────────────────────────────────────

    def create_ui(self, api_session):
        with gr.Column() as root:
            if self._patch_error:
                gr.Markdown(f"⚠️ **RefMods could not hook into MiniMax H3**: {self._patch_error}")
            gr.Markdown(f"## {PlugIn_Name}\n"
                       "No-training reference mods for MiniMax H3 Ref2VA. See the README bundled "
                       "with this plugin for how the mechanism works and its current limitations.")
            model_choices = _model_choices(api_session)
            model_dd = gr.Dropdown(
                label="MiniMax H3 Ref2VA model (used for Extract and Generate below)",
                choices=model_choices, value=_default_model_choice(model_choices))
            with gr.Row():
                refresh_models_btn = gr.Button("Refresh model list", size="sm")
                diagnose_btn = gr.Button("Check setup for this model", size="sm")
            diagnose_status = gr.Markdown("")
            refresh_models_btn.click(fn=lambda: gr.update(choices=_model_choices(api_session)),
                                     outputs=[model_dd], queue=False)
            diagnose_btn.click(fn=lambda mt: _diagnose(api_session, mt, self._patch_error),
                              inputs=[model_dd], outputs=[diagnose_status], queue=False)

            with gr.Tabs():
                with gr.Tab("Extract"):
                    self._build_extract_section(api_session, model_dd)
                with gr.Tab("Library"):
                    self._build_library_section()
                with gr.Tab("Generate"):
                    self._build_generate_section(api_session, model_dd)
        return root
