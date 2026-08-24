# MiniMax H3 RefMods -- a Wan2GP plugin

No-training "reference mods" for MiniMax H3 **Ref2VA**: compress an image or
video reference into a small `.safetensors` file once, then reuse it in later
generations at any strength -- without re-encoding the original picture/clip
every time, and without needing to keep the original file around at all.

This is a port of the idea and math from
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
(MIT License, (c) 2026 Luisa/luisacaotica) onto [Wan2GP](https://github.com/deepbeepmeep/Wan2GP),
which already ships its own native MiniMax H3 implementation using the exact
same 24-channel video VAE. `core.py` in this plugin is close to a direct port
of that project's `core.py` (which has no ComfyUI-specific code at all, so it
travels almost unchanged); everything else (storage, the UI, and the
generation hooks) is new, written specifically for Wan2GP's plugin API and
its own `MiniMaxH3Pipeline`.

**Mods made by either tool are interchangeable** -- they're the same VAE
latent in the same `.safetensors` layout, so a mod extracted in ComfyUI can be
dropped into `loras/refmods_plugin/minimax_h3/` here and used directly, and vice versa.

## Two ways to use RefMods

**Recommended: the inline panel on Wan2GP's own Media Generator page.** Once
a MiniMax H3 Ref2VA model is selected there, an extra **"MiniMax H3 RefMods
(inline)"** accordion appears (near the LoRAs Multipliers field). Pick up to
9 image + 2 video mods with a strength each and hit *that page's own*
Generate button as usual -- every native field (resolution category/budget,
frame count and its duration in seconds, steps, sampler, the full
attention-mode list, memory profile, FPS override, output filename
template, text encoder/VAE variant, DiT priority, LoRAs, sliding window --
everything) works completely unchanged, because none of it is duplicated:
this plugin only adds the RefMod picker itself, on the same page. See "How
the inline panel works" below for the mechanism and its one caveat.

**Alternative: this plugin's own Extract / Library / Generate tab.** Useful
if you'd rather not touch the main form, or want a second, independent
generation queue. Its Generate section covers the fields people tune most
often directly, plus a **Sync from the main form** button that copies every
other current setting from the Media Generator tab in one click.

Either way, extraction always happens from this plugin's own **Extract**
tab.

## Why a reference mod at all

MiniMax H3's reference path (`Ref2VA`) works by VAE-encoding your reference
image/video, patchifying it, and letting every block of the transformer
attend to those tokens. A full-resolution video reference is expensive: it
can contribute thousands of tokens *every single generation*. A RefMod is the
same reference, saved once:

- **`training` mode (default)**: the encode is average-pooled down to a tiny
  grid (e.g. 16x16 = 64 tokens/frame) and optionally refined with a few
  gradient steps against the full encode (a handful of seconds, no diffusion
  model involved -- this is the only "training" happening anywhere). Cheap,
  carries concept/style/motion well, softer on fine identity at small grids.
- **`encode` mode**: the full-resolution VAE encode is kept as-is (same cost
  as a live reference, but you only pay the encode once and can reuse it
  forever). Best for identity/character fidelity.

At generation time the saved latent is handed back into the exact same
reference-conditioning path a live image/video reference would use, so it
goes through the same attention machinery -- only the token count changes.

## Installation

1. Copy this whole folder into your Wan2GP `plugins/` directory, e.g.:
   `plugins/wan2gp-minimax-h3-refmod/`
2. Start Wan2GP, open the **Plugins** tab, enable **MiniMax H3 RefMods**,
   save settings, and restart Wan2GP (standard Wan2GP plugin flow).
3. A new **MiniMax H3 RefMods** tab appears with three sections: **Extract**,
   **Library**, and **Generate**.

No extra Python dependencies beyond what Wan2GP already ships (torch,
safetensors, numpy, Pillow). `requirements.txt` only lists an optional video
decoding backend as a nice-to-have.

You may notice two extra, normally-empty text boxes named "RefMods selection
(managed by the MiniMax H3 RefMods plugin -- leave blank)" and "RefMod
extraction job (...)" under **Custom Settings** on Wan2GP's own Media
Generator tab whenever a MiniMax H3 Ref2VA model is selected. That's
expected -- see "How the injection actually works" below for why they exist.
Leave them blank; this plugin's own Extract/Generate buttons fill them in on
submission.

## Using it

Before extracting or generating, you can click **Check setup for this
model** (next to the model dropdown, in this plugin's own tab) to confirm
the plugin's hooks are actually active for the model you selected -- it
reports whether the pipeline patch loaded correctly and whether that model
declares the two `custom_settings` ids RefMods rely on. If either check
fails, extraction/generation will silently fall back to a normal render
instead of doing what you asked, with nothing else in the log to explain why
-- so this is worth a quick check the first time, or after updating Wan2GP.

### The inline panel (Media Generator page)

Select a MiniMax H3 Ref2VA model, then open the **"MiniMax H3 RefMods
(inline)"** accordion. Pick mods, set strengths/copies, optionally a master
retention and a curve, then just use the page's own Generate button --
nothing else to configure, nothing else changes. Selections here apply
immediately and don't require a separate "Apply" step, and don't affect
anything if a non-MiniMax-H3 model is selected instead.

### Extract

Upload one or more reference images and/or up to two reference videos (the
same "Use Two Reference Videos" input MiniMax H3 Ref2VA natively supports),
name the mod, pick `training` or `encode` mode, and hit **Extract & Save
RefMod**. Every reference you provide -- images and video(s) alike -- gets
encoded and stacked into this one mod file. This briefly runs as a real
(near-instant) generation task on the MiniMax H3 Ref2VA model you selected,
so it can reuse that model's already-loaded VAE instead of this plugin
trying to load weights on its own. **You will see the normal generation
progress bar for a few seconds and then no video will appear -- that's
expected.** The mod was still saved; check the status box under the button,
or the Library tab, to confirm.

**Mode is the single most consequential choice here** -- it controls what
actually ends up in the saved tensor, unlike the metadata-only fields below:
- **`encode`**: the full VAE-encoded latent is kept as-is, at full fidelity
  -- identical to what a live reference would produce. Best for a person's
  precise face/identity. Bigger file, more tokens spent at generation time
  (same cost as a live reference), and the "Identity refinement steps"
  option (Advanced) is ignored -- there's nothing left to refine.
- **`training`**: the latent is average-pooled down to a tiny grid (the
  "Pool grid" sliders, Advanced) and optionally refined with a few hundred
  gradient steps against the full encode ("Identity refinement steps",
  Advanced -- optimizing only the small saved latent's own values, no model
  weights involved). Pooling is genuinely lossy: fine detail (eyes, nose,
  mouth proportions) gets averaged away, which is fine for a general
  concept/style/pose but is exactly what causes a specific face to drift
  toward a generic "chubbier/older" look -- hence the live warning under the
  pool grid sliders when `concept_type=identity` is combined with a small
  grid. Smaller file, fewer tokens at generation time.

**Video reference duration to use (seconds)** shows and sets this directly
in real seconds of the source video (0.1s steps) rather than the abstract
"latent frames" count the underlying extraction actually works in -- so if
your video is 4 seconds long, you can just drag the slider to 4.0s instead
of guessing what frame-count number that corresponds to. It goes up to
14.9s -- MiniMax H3 Ref2VA's own 15-second reference cap works out to about
90 latent frames once the video VAE's causal 4:1 temporal compression is
accounted for (verified: 90 latent frames ≈ 357 pixel frames ≈ 14.9s at
24fps; the next notch up would already be over 15s). This is a **shared**
budget: if a generation combines two video-kind mods (the two "Video Mod"
slots), their durations *add together* against that same 15s ceiling, so a
single mod extracted near 14.9s leaves no room for a second one alongside
it -- the live counter (Generate tab / inline panel) always shows the real
total for whatever's currently selected.

A source video always contributes a **contiguous** clip from its start,
matching the slider as closely as the video VAE's own frame grid allows --
in both modes. (An earlier version of this plugin sampled `encode` mode's
frames sparsely across the *entire* source video instead of a contiguous
prefix; the video VAE has no idea a "frame" was pulled from 8 seconds in
rather than 0.3 seconds in, so it compressed the scattered sample as if it
were a real, short, sequential clip -- silently breaking the slider's
promise. Fixed.)

Two things can still make the *saved* mod end up shorter than what you
asked for -- extraction always tells you plainly when this happens, with
the requested vs. actual duration:
- In `training` mode, the slider is a **ceiling, not a guarantee**: pooling
  can't invent seconds the source video didn't produce enough of once
  VAE-compressed, so a short source video ends up shorter than requested no
  matter how high the slider is set.
- **`Max tokens`** (Advanced, defaults to 65536 -- unlocked, so it stays out
  of the way for most extractions) applies *after* encoding, and `encode`
  mode at a high "Ref resolution" can still burn through even that: a
  1024px `encode`-mode video ref costs 1024 tokens per frame, so 65536
  tokens fits about 64 frames (~10.6s). Past that point -- or with a lower
  budget set deliberately -- the mod still gets saved, just shorter.

**Automatic background removal (optional).** Same "Automatic Removal of
Background behind People or Objects in Reference Images" toggle Wan2GP's own
Media Generator form offers for reference images -- same `rembg` call and
alpha-matting parameters, so a mod extracted with this on looks consistent
with a live reference image processed the same way. Defaults to **Keep
Backgrounds behind all Reference Images**. Only applies to the reference
image(s), not to reference videos, matching Wan2GP's own behavior. Applied
in pixel space before encoding, so it works the same in both `training` and
`encode` mode.

**`keyword - description` and `Concept type`** (under Advanced) are the
opposite of Mode: **pure metadata with no effect on generation whatsoever**.
Nothing in a mod file is ever read by the model automatically -- MiniMax
H3's Ref2VA path has no image-embedding / CLIP-Vision-style channel to hang
an automatic clue off of, so text typed into your prompt is the only
channel that actually reaches the model. They only become useful through
the Library tab's **Build Prompt Hint** tool (or by copying them into your
prompt yourself):
- `keyword - description`, written the same way you'd write a LoRA training
  caption, is the text that gets used.
- `Concept type` just prefixes it in that output (e.g.
  `identity: ginger woman, tattooed neck`), the same way the original
  ComfyUI pack's loader does -- and also drives the live pool-grid warning
  mentioned above.

### Library

Lists every saved mod (name, kind, mode, token count, file size, description)
read straight from disk. Delete mods you no longer need. Mods live in
`loras/refmods_plugin/minimax_h3/` at the root of your Wan2GP install.

**Fix classification.** Mods extracted purely from several still images (no
video source) were, before this fix, wrongly saved as `video` kind whenever
more than one image was stacked together into the same mod -- because the
underlying latent ends up with more than one frame either way, and the
original logic used "more than one frame" as its only signal instead of
checking whether any of the sources was an actual video. A `video`-kind mod
gets injected as *one* temporal/motion reference (through one of only 2
video slots), while several still images should really be *several*
independent identity references (through the 9-image path) -- so a
misclassified mod could both waste a scarce video slot and get
misinterpreted as a mini "animation" between unrelated photos instead of
several separate looks at the same subject. Click **Fix classification** to
rescan every saved mod's own extraction record and correct this in place --
only the `kind` label changes, the latent data itself is never touched, and
it's safe to run repeatedly (already-correct mods are left alone). New
extractions made with this version already classify correctly from the
start.

**Rename or edit a mod's description.** Pick a mod from the dropdown, click
**Load** to pull its current name/description into the two text boxes below,
edit either one, then **Save changes**. Renaming re-saves the file under the
new name and removes the old one (refused if a mod with that name already
exists, so you never lose one by accident) -- the latent data is byte-for-byte
untouched either way. Both fields are plain editable text boxes, so clicking
into one and selecting the text (double/triple-click, or Ctrl+A) then Ctrl+C
copies it like any other text on the page -- no separate copy button needed.

### Generate (this plugin's own tab)

**A large banner at the top of this tab points you to the Media Generator
tab's inline panel instead** -- see "Two ways to use RefMods" near the top
of this document. This tab is kept as a simpler, self-contained fallback.

A generation form covering the fields people tune most often: prompt,
resolution, number of frames, seed, number of inference steps, flow shift,
sampler, reference-image pixel budget, and up to 9 image-kind + 2 video-kind
RefMod slots (mod +
strength + copies), plus a master **retention** dial and an optional
strength **curve** under a small accordion. An **Advanced Mode** accordion
adds:

- **LoRAs** -- pick from the LoRAs installed for the selected model, plus a
  multipliers text field (same syntax as the main form).
- **Steps Skipping** -- Spectrum Feature Forecasting / First Block Cache,
  with the threshold and starting-percentage controls.
- **Sliding Window** -- window size / overlap, used automatically once the
  frame count needs more than one window.
- **Attention** -- override attention backend (e.g. Sol-Attn) and its Start
  Tau sparsity dial.

Click **⟲ Sync from the main form** first to copy every other current
setting from Wan2GP's own Media Generator tab (for whichever model is
selected there) as your starting point -- including anything this panel
doesn't have a dedicated widget for (audio references, live reference
images/video, output filename template, memory profile, text encoder/VAE
variant, category/resolution budget as a single string, etc.). Whatever you
don't sync or override here still gets a valid value from that model's own
factory defaults before submission, so nothing is ever left unset -- though
if you want every native field editable as its own widget rather than
inherited from a snapshot, the inline panel above is the better fit.

## How the inline panel works (and its one caveat)

Wan2GP's own "Custom Settings" fields (the channel this plugin uses to carry
a RefMod selection to the pipeline, see below) are auto-built from a model's
definition but never given a stable `elem_id`, so a plugin cannot bind its
own rich widgets to them directly -- there is no way to make the inline
panel's mod pickers *be* one of those fields. Instead:

- The panel is injected via `insert_after("loras_multipliers", ...)` -- the
  only elem_id Wan2GP's plugin docs guarantee is present on (almost) every
  model's form, so the panel can be added once, at UI-build time, without
  needing per-model wiring.
- Visibility is handled separately, by binding to a hidden text component
  Wan2GP itself already updates on every model switch. Note that Wan2GP
  hands plugins their requested components out of `generate_media_tab`'s
  own `locals()`, so the key is the Python **variable** name
  (`model_choice_target`) -- *not* its elem_id
  (`wangp_model_choice_target`), which is what an earlier version of this
  plugin wrongly requested, silently leaving the panel always-visible. Both
  names are requested now, for compatibility across builds. Its value
  (`"{model_type}|{timestamp}"`) is parsed the same way Wan2GP's own
  `_model_choice_target_model_type` does, and the panel is shown only when
  the model type starts with `minimax_h3_ref2va`. If neither name resolves,
  the panel falls back to always-visible rather than silently disappearing
  -- it still has no effect on any other model either way, this only
  changes whether it's shown.
- Wan2GP auto-saves the whole form continuously as you edit any field (via
  `save_inputs`/`prepare_inputs_dict`), which is also what the real Generate
  button ultimately reads from. So instead of writing into that
  continuously-rebuilt settings dict directly (which the very next field
  edit would silently wipe, since none of the panel's widgets are native
  form fields Wan2GP's own save logic knows about), the panel writes its
  JSON payload into a namespaced key on the session `state` dict this
  plugin owns (`state["_h3refmod_selection"]`) that nothing else in Wan2GP
  ever touches.
- `prepare_inputs_dict` itself is wrapped (via `self.set_global`, Wan2GP's
  own supported way for a plugin to replace one of its globals) so that,
  *every* time it runs -- including the one that matters, right when
  Generate is clicked -- it re-reads that stashed key and folds it into that
  call's `custom_settings`. This survives the autosave cycle by construction,
  since it re-injects on every single call rather than being overwritten by
  one. The wrapper calls the original function unchanged first and only adds
  to its result for MiniMax H3 Ref2VA model types with a non-empty
  selection stashed -- every other model, and MiniMax H3 with nothing
  selected, behave exactly as before.

**The caveat**: this patches a function used by every model in Wan2GP, not
just MiniMax H3 -- a much larger blast radius than this plugin's other
patches, even though the added logic is narrowly scoped and wrapped in its
own `try/except` (a failure here logs a warning and leaves the original
result untouched, it never raises). If you notice *anything* unusual with
non-MiniMax-H3 generations after enabling this plugin, that's the first
place to look, and disabling the plugin removes the patch entirely (nothing
is written to disk). The plugin's own Extract/Library/Generate tab does not
depend on this patch at all and is unaffected either way.

## How the RefMod injection itself works (for anyone auditing this)

Wan2GP's `MiniMaxH3Pipeline.generate()` builds a `refs` list (kind/shape
metadata) and a `visual_latents` list (the actual VAE latents) from whatever
live references you pass it, via two internal helpers,
`_add_image_reference` and `_add_video_reference`. Those lists are local to
`generate()`, so a plugin outside Wan2GP's own source tree cannot reach into
them directly -- and reimplementing that ~300-line method here would be
fragile and guaranteed to drift out of sync with upstream.

Instead, `patches.py` applies four small, targeted monkeypatches to the
already-imported `MiniMaxH3Pipeline` / `family_handler` classes at plugin
load time:

1. `_add_image_reference` / `_add_video_reference` are wrapped to recognize a
   tiny sentinel object carrying a precomputed latent; when they see one they
   append it straight into `refs` / `visual_latents` (skipping the pixel
   resize + VAE encode a live reference goes through), and otherwise fall
   through to the original, unmodified implementation.
2. `generate()` is wrapped so that, just before calling the original, it
   reads a small JSON blob out of Wan2GP's own generic `custom_settings`
   channel (already plumbed end-to-end from a submitted task to
   `pipeline.generate(**kwargs)` for several other models) and turns it into
   sentinel objects appended to `input_ref_images` / `input_frames` /
   `input_frames2` -- the same public parameters a live reference uses. This
   is what lets a RefMod apply even when you supply *no* live reference at
   all, which is the entire point of the feature.
3. The same wrapper also recognizes a second `custom_settings` key that means
   "this call is a RefMod extraction, not a real render": it runs the VAE
   encode/compression directly and returns `None` immediately, which is
   exactly what `generate()` already does when a user aborts mid-generation
   -- so nothing downstream needs to change to handle it gracefully.
4. `family_handler.query_model_def` is wrapped to add two "text"
   `custom_settings` entries (ids `h3_refmod_state` / `h3_refmod_extract`,
   the two keys used above) to MiniMax H3's model definition. **This one
   isn't optional.** Wan2GP validates every submitted task's
   `custom_settings` against the ids the target model declares (via
   `collect_custom_settings_from_inputs`, called from `validate_settings`
   for *every* task, including ones submitted through the API/plugin path)
   and silently replaces anything else with `None` -- so without this,
   points 2 and 3 above would build a correct payload that then gets wiped
   out one step later, with no error anywhere to explain why generation
   just... ran normally instead of doing what was asked. (This was exactly
   the bug in the first published version of this plugin: extraction quietly
   ran a full render instead of saving a mod, with nothing in the logs
   pointing at the real cause.)
5. Wan2GP builds its entire model catalog (`models_def`, what
   `get_model_def()` reads from -- a plain dict) **once, at import time,
   before any plugin is loaded.** Patching `query_model_def` (point 4) has
   no effect on entries computed before the patch existed -- it only affects
   *future* calls. So `plugin.py`'s `post_ui_setup` (which runs once
   Wan2GP has finished injecting plugin globals, still before the app starts
   serving requests) calls Wan2GP's own `refresh_model_defs()` -- its
   supported way to rebuild that catalog on demand -- once, forcing every
   MiniMax H3 model definition to be recomputed through the now-patched
   function. (This was the second bug found while fixing the first one: the
   `query_model_def` patch alone was necessary but not sufficient, since it
   never got a chance to run before the catalog it was supposed to affect
   had already been built.)
6. `prepare_inputs_dict` is wrapped (also via `post_ui_setup`, using
   `self.set_global`) to make the *inline* Media Generator panel work --
   see "How the inline panel works" above for the full explanation. Its
   patch is scoped as narrowly as possible (only touches its result for
   MiniMax H3 Ref2VA model types with a stashed selection) and wrapped in
   its own `try/except` (any failure leaves the original result untouched).
7. `_as_video` (a plain module-level helper in `pipeline.py`) is wrapped so
   it passes a video-kind RefMod sentinel through unchanged instead of
   crashing on it. `generate()` runs every entry of `input_frames`/
   `input_frames2` through this function itself, *before* looping over them
   to call `_add_video_reference` -- so the sentinel has to survive this
   call too, not just the one inside `_add_video_reference`, or it fails
   with `'_RefModVideoSentinel' object has no attribute 'ndim'` before point
   1's patch ever gets a chance to run.
8. Right after that, `generate()` also computes a total-duration budget
   check (`sum(video.shape[1] for video in video_sources) / fps`, enforcing
   Ref2VA's 15-second reference cap) before ever reaching
   `_add_video_reference`. The video-kind sentinel exposes a `.shape`
   property for exactly this -- a plausible reconstructed pixel-space shape
   derived from the latent's own dimensions (undoing the video VAE's causal
   4:1 temporal compression and 16x spatial downsampling), not real pixel
   data, since a RefMod has none to offer.
9. `validate_generative_settings` (a third staticmethod on the same
   `family_handler` class as point 4, patched alongside it) is wrapped so an
   audio reference doesn't get rejected as "0 visual references" just
   because the visual side is coming entirely from RefMods. This check runs
   *before* `pipeline.generate()` is ever called, straight off the raw
   native form fields (`image_refs`/`video_guide`), with zero visibility
   into RefMods -- so without this, combining an audio reference with
   RefMod-only visuals (no live reference image/video) would always be
   rejected by this pre-flight check, even though the RefMods would have
   supplied enough visual references once actually injected. The patch
   calls the original function first, unchanged, and only steps in for this
   one specific failure message -- recounting visual references with
   RefMods included and clearing the error if that's now enough; every
   other rule the original function enforces (durations, per-type caps,
   control-video-specific checks) is left completely untouched.
10. `_resize_video` (another module-level helper in `pipeline.py`, present
   in newer Wan2GP builds) is wrapped to pass a video-kind sentinel through
   untouched. Newer versions resize each reference video to the output
   resolution right before `_add_video_reference`; a RefMod carries an
   already-VAE-encoded latent rather than pixels, so there is nothing to
   bicubic-resize (and no pixel tensor to `.permute()`) -- the latent goes
   into the packed sequence at its own saved resolution. On older builds
   without this helper the patch is simply skipped.
11. `get_model_settings` (a plain wgp.py function, patched via
   `self.set_global` like `prepare_inputs_dict`) closes a gap that one
   doesn't cover: clicking **Generate** on the Media Generator page doesn't
   call `prepare_inputs_dict` again -- it reads the task straight out of
   `get_model_settings(state, model_type)`, a cache last refreshed whenever
   `save_inputs`/`prepare_inputs_dict` most recently ran, which only
   happens when a *native* form field changes. If the last thing touched
   before clicking Generate was a RefMod picker in the inline panel and
   nothing else, that cache could be one selection behind. This wraps
   `get_model_settings` to re-apply the same freshest-`state[STASH_KEY]`
   injection one more time, right at the point the task is actually
   assembled -- the last possible moment before it's queued.

None of this edits any file inside your Wan2GP install; it's applied purely
in-memory, once, and is safe to apply twice (idempotent) if the plugin is
reloaded.

## Known limitations / not (yet) ported

Compared to the ComfyUI pack, this version does **not** include:

- The A/B "axis" loader (two mods on one signed slider).
- Bulk folder extraction.
- Saved/shareable curve-graph PNG presets.
- The per-denoising-step curve wrapper (`H3 RefMod Step Curve`) -- only the
  per-frame curve (baked in once, at injection time) is available here.
- More than 2 simultaneous **video-kind** mods in one generation, and at most
  9 **image-kind** mods, at most 12 total -- these are Wan2GP's own native
  Ref2VA reference limits, not an extra restriction added by this plugin.
  **A mod extracted from several stacked images counts once per image it
  contains**, not once per mod: a mod trained on 5 images uses 5 of the 9
  available image slots by itself. This isn't a soft cap this plugin could
  safely raise -- MiniMax H3's own packing code assigns a single-frame
  position grid to every "image" reference, so feeding it more than one
  frame either crashes or leaves every frame at the exact same position
  (silently indistinguishable to the model), rather than degrading
  gracefully. A live counter above the mod pickers (**Images: X / 9**,
  **Video: ~Xs / 15s**) tracks this in real time, accounting for strength,
  copies, and each selected mod's own frame count, so you don't have to do
  the math by hand or discover an overflow only when generation fails.
- The inline panel (and this plugin's own Generate tab) offer 9 image-kind +
  2 video-kind mod slots -- matching MiniMax H3 Ref2VA's own native caps on
  each, so there's no situation where a slot is available but couldn't
  possibly be used.
- The inline panel hides automatically after a model switch (see "How the
  inline panel works" below), but stays visible right after Wan2GP starts
  if a MiniMax H3 Ref2VA model was already selected and nothing else has
  been switched yet -- there's no confirmed signal for "the page just
  loaded with model X" to react to, only for an actual switch.

Each selected video-kind mod goes into its own native "Reference/Control
Video" slot -- the exact same mechanism MiniMax H3 Ref2VA's own "Use Two
Reference Videos" option uses, one mod per slot -- rather than being merged
into a single combined tensor. Since there are only 2 such slots, at most 2
video-kind mods can be used per generation (matching the 2 "Video Mod" rows
in the picker); if a live reference video from the main form already
occupies both slots, video-kind mods have nowhere left to go and generation
falls back to running without them (logged, not a hard failure). The two
slots are fully independent -- mods of different resolutions can be used
together without issue.

## Using a saved RefMod

Each of the up to 9 image + 2 video slots in the picker is just a mod name
and a **Strength** slider (0 to 2, default 1). That's the whole surface:
retention, curve, scramble-seed, and per-mod copies were part of earlier
versions of this plugin and have since been removed -- in testing they
added UI complexity without changing outcomes enough to be worth it for
most people, and a strength of 1 already means "use the mod as saved".
Strength above 1 doesn't clip at the mod's original encode -- it extrapolates
past it, which can push a subtle mod harder but can also push it into
artifacts; there's no ceiling built in beyond the slider's own 2.0 max, so
treat values past 1 as an experiment, not a default.

## Testing notes (please read before relying on this)

This plugin was written and unit-tested against Wan2GP's source code
directly, including a fully mocked stand-in for `MiniMaxH3Pipeline` that
exercises the exact extraction and injection code paths above end-to-end.
**It has not been run against the real ~33B/20B MiniMax H3 weights on a GPU**,
since that isn't possible in the environment this was written in. Please
verify, on your own machine, before relying on it:

- That extraction actually produces a plausible reference in a follow-up
  generation (compare a `strength=1.0`, `mode=encode` mod against feeding the
  same image as a live reference -- they should look close to identical).
- That mixing several mods, and the `retention`/curve controls, behave the
  way the tooltips describe.
- The console/log output during extraction and generation, if anything looks
  off -- every step in `patches.py` prints a `[H3RefMod] ...` line.

If `models.minimax_h3.pipeline` has moved or changed shape in your installed
Wan2GP version, `install_patches()` fails safe: it prints a clear message and
the plugin's tab still opens (with extraction/generation disabled) instead of
crashing Wan2GP's startup.

You may see the `[H3RefMod]` setup lines (patched pipeline, refreshed model
catalog, hooked `prepare_inputs_dict`) printed twice at startup -- Wan2GP
appears to call plugins' `post_ui_setup` more than once during its own
startup sequence. This is harmless (every patch here checks a marker before
re-applying itself) and only costs a fraction of a second rebuilding the
model catalog an extra time; it doesn't indicate anything went wrong.

## License / attribution

This plugin's `core.py` is a close port of `core.py` from
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
by Luisa (luisacaotica), MIT License, (c) 2026. The rest of this plugin
(`storage.py`, `patches.py`, `plugin.py`) is new code written for Wan2GP.
