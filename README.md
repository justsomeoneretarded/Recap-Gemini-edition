# Recap Shorts — Gemini edition

A movie recap Shorts pipeline with its own **Generate Shorts** workflow. Choose footage, a story brief, a narrator, a maximum duration and square or vertical output. The Gemini edition adds a **multi-stage, evidence-grounded Gemini pipeline**. Gemini watches the source footage, builds a timestamped event inventory and story map, chooses the story, writes and reviews the script, matches every narration sentence to footage that shows it, reviews the edit independently and repairs specific problems. Rendering is still deterministic FFmpeg. Voices (Chatterbox), source dialogue (Whisper), captions and loudness handling are the existing, tested components. See the [architecture](docs/gemini-architecture.md).

The style profile comes from eight owner-supplied references, analysed in [reference analysis](docs/reference-analysis.md). Its main features are action before backstory, causal story beats, compact white italic captions, meaningful object/face shots, selected movie dialogue and story-dependent endings. The references are 1080×1080 and 136–180 seconds long. Square is the default and vertical (9:16) is an option. This is an engineering implementation, not a verified recreation of the creator's process. **The Gemini path has been tested offline only (mocked API). It has not yet produced a real recap with a live key.** See [validation](docs/validation.md).

## 1. Get a Google AI Studio API key

1. Open [Google AI Studio](https://aistudio.google.com/) and sign in with a Google account.
2. Choose **Get API key → Create API key** and pick or create a Google Cloud project.
3. **The free tier works**: the default `gemini-free` preset is built for it and costs nothing. Free keys have per-minute and per-day request limits, and some models (typically Pro) may have no free quota. The pipeline paces its requests, falls back to the next available Flash/Flash-Lite model, and stops cleanly if the daily quota runs out. Google may use free-tier prompts and uploaded footage to improve its products (see the Gemini API terms), so do not upload footage you are not allowed to share. The paid presets need a billing-enabled project.
4. Copy the key. Treat it like a password: never paste it into an issue, a commit or a workflow input.

## 2. Add the key to GitHub

1. In this repository open **Settings → Secrets and variables → Actions**.
2. Choose **New repository secret**, name it **`GEMINI_API_KEY`**, paste the key and save.
3. Optional: under the **Variables** tab, add `GEMINI_PRICES_JSON` with current list prices from the Gemini pricing page. This makes the cost report exact rather than conservative, for example:
   `{"gemini-3.8-flash": {"input": 0.5, "output": 3.0, "cached": 0.05, "audio": 1.0}}`

The workflow passes the secret only to the key check and the generate step. It is never echoed, and error messages redact it.

## 3. Generate a Short on GitHub

Open **Actions → Generate Shorts → Run workflow** and fill in:

| Field | What to provide |
|---|---|
| Title | Your Short's title |
| Source URL | A direct HTTPS URL to clean movie/scene footage (GitHub Release attachments work) |
| Brief | The story angle and editing instructions |
| Story notes | Verified names and context. Names are only narrated if they appear here, in clearly spoken dialogue or on screen |
| Narrator | Warguy, Kit Laser or Female |
| Maximum seconds | 15–180. Use 135–180 for the reference style; default 150 |
| Aspect | Square or vertical |
| Preset | `gemini-free` (default; free AI Studio key), `gemini-balanced`, `gemini-quality`, `gemini-economy` (paid tier), or `local-gemma4` (original offline pipeline; no key needed) |
| Source window | Optional `start,end` seconds of the relevant excerpt, e.g. `51,73` |
| Options | Optional extras, see below |

**Options** are `key=value` pairs separated by spaces:

| Option | Meaning | Default |
|---|---|---|
| `rpm=8` | Free tier: Gemini requests per minute (lower it if you see rate-limit retries) | 8 |
| `max_requests=150` | Free tier: request cap per run, protecting the daily quota | 150 |
| `budget=3` | Paid presets: maximum estimated spend in USD | quality 8 · balanced 3 · economy 1 |
| `review_passes=2` | Independent review → targeted-correction rounds (0–4) | free 1 · quality 2 · balanced 2 · economy 1 |
| `analysis=detailed` | Video analysis quality: `fast`, `standard`, `detailed` | free standard (≤6 zoom checks) · quality detailed · balanced standard · economy fast |
| `fallback=gemma4:12b` | Use the local writer if Gemini is unavailable (adds Ollama setup time) | none |
| `speech=turbo` | Chatterbox model: `original`, `turbo`, `nano` | original |
| `alignment=auto` | Refine caption word timing with MMS forced alignment (≈1 GB model download) | off |
| `strict=true` | Fail instead of rendering when critical review issues remain | false |
| `writer=qwen3:4b-instruct-2507-q4_K_M` | Local writer for the `local-gemma4` preset | gemma4:12b |

The job page shows a summary: state, review status, rendered-file checks, and estimated cost per stage. Download **Artifacts → shorts-N** for:

| File | Contents |
|---|---|
| `final.mp4` | The rendered Short |
| `script.txt`, `script-draft.json`, `description.txt` | Final script; draft, review notes and evidence references |
| `event-inventory.json` | Timestamped event inventory with verification status and rejected events |
| `story-map.json`, `editorial-plan.json` | Story map (observed vs inferred) and the editorial selection |
| `evidence-map.json` | Sentence → shot evidence: what each shot shows, why it was placed, in/out, anchor |
| `storyboard.json`, `timeline.json` | Storyboard and measured, frame-allocated timeline |
| `narration-timing.json`, `captions.srt/.ass`, `caption-timing.json` | Narration and word timing, captions, timing provenance |
| `quality-review.json`, `correction-history.json` | Independent review issues with severity, and every correction attempt |
| `final-verification.json`, `quality-report.json` | Measurements of the rendered file; existing render QC |
| `cost-report.json`, `gemini-settings.json` | Models per stage, tokens, cache use, estimated cost, retries, timings, preflight estimate |
| `cover-01..03.jpg`, `contact-sheet.jpg`, `source-attribution.json` | Cover candidates, contact sheet, source provenance |

`review_status` is `passed_automated_review`, `passed_with_warnings` or `needs_human_review`. **No result is marked production-ready**: watch it with sound before publishing. There is no automatic upload.

### Free tier: what to expect

- **Cost:** $0. `cost-report.json` still lists tokens and the *paid-price equivalent* for comparison.
- **Request use:** a typical 150-second Short needs roughly 25–45 requests, paced at 8 per minute. The preflight estimate is in `gemini-settings.json`.
- **Speed:** pacing adds a few minutes. Chatterbox narration on the CPU runner is still the slowest step.
- **Hitting a limit:**
  - Per-minute limits are retried after the wait the server asks for.
  - Daily quotas are **per model**, and some free models allow only a handful of requests per day. When a model's daily quota runs out, the run switches to the next Flash/Flash-Lite candidate and remembers the exhausted model until the quota resets (midnight Pacific time).
  - Only when every candidate is used up does the run stop with a clear message. Finished stages are saved in the Actions cache, so running the same inputs again later resumes where it stopped.
  - A model with no free quota is skipped automatically in favour of the next candidate, and the switch is recorded.
- **Quality:** free-tier quality depends on which Flash model your key can use. Pro-class writing/review models usually need the paid presets.

### Estimated Gemini cost (paid presets)

These figures come from the pipeline's **pre-flight estimator**, which uses upper-bound token assumptions and is not a measured bill. For a 150-second Short:

| Preset | 3-min excerpt | 10-min excerpt | 30-min excerpt |
|---|---|---|---|
| economy (2.5 list prices / newest-model placeholders) | $0.15 / $0.45 | $0.17 / $0.50 | $0.23 / $0.71 |
| balanced | $0.57 / $1.41 | $0.67 / $1.69 | $0.96 / $2.57 |
| quality | $1.85 / $4.71 | $2.39 / $6.11 | $3.69 / $9.63 |

The first number uses published Gemini 2.5 list prices. The second uses deliberately high placeholder prices for newer models whose prices are not in this repository. Actual usage is usually lower than these upper bounds. Each run records the real token counts in `cost-report.json`, and Google AI Studio billing is authoritative. Reruns of unchanged stages are served from cache at no cost.

## How the Gemini pipeline works (short)

- **A — Footage understanding:** windowed native video analysis with burned-in timecode, then code checks, then zoom re-inspection of uncertain, fast or important events.
- **B — Story map:** observed events kept separate from inferences, which must cite their evidence.
- **C — Editorial selection:** hook, causal chain, dialogue clips and ending mode.
- **D — Script (hybrid by default):**
  - Gemini cuts clips from the chosen events and writes a footage-first script line by line for those clips (Script A).
  - It also writes a story-first script with context (Script B), followed by a separate review pass.
  - A merge pass keeps A's lines on their exact clips and adds B's context over unused related cover footage.
  - Each line's key word is timed to the moment its clip shows the action.
- **E — Matching:** Gemini watches each sentence's candidate clips and picks in/out points and the action anchor.
- **F — Shot plan:** deterministic placement with no footage reuse.
- **G — Sync:** measured Chatterbox audio sets per-sentence cuts; overflow and footage shortfalls trigger targeted rewrites.
- **H — Review:** an independent review that watches the placed shots.
- **I — Correction:** bounded, targeted fixes.
- **J — Verify:** checks on the rendered MP4.

Details are in [docs/gemini-architecture.md](docs/gemini-architecture.md).

**Publishable output needs clean footage.** The analysis stage reports when a source has burned-in captions or voice-over narration. Such a run is marked `needs_human_review` with a source warning on the job page, and retained "dialogue" clips are disabled so an old narrator is never presented as a film character. Several other measures keep a Short from running out of picture:
- Clips are widened to cover their whole verified event.
- The narrator's real speaking speed is measured and saved for later runs.
- An underlength Short is extended only into footage it already has.
- Footage is never reused.

So a 150-second Short needs roughly 4–6 minutes of clean source.

Use clean movie/scene footage. The eight reference Shorts already contain narration and burned-in captions, so they are style references and test material, not production sources. A movie title alone cannot supply footage. For full films use **Source window**: Gemini analysis refuses sources longer than 45 minutes unless `max_source_minutes` is raised in a CLI request. The local preset keeps its original 120-span cap. YouTube downloading uses yt-dlp but can be blocked on runners; direct media URLs are reliable.

## Local setup

Use Python 3.11 or 3.12, FFmpeg/FFprobe with libass, and DejaVu fonts. Keep this environment separate from Graphify's environment because their dependency versions differ.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[voice,analysis,source,gemini,test]'
```

For the Gemini pipeline, put the key in your shell environment, never in a file you commit:

```bash
export GEMINI_API_KEY='paste-your-key-here'
```

```bash
python -m shorts_pipeline generate --request examples/request.json --output output/shorts
```

`examples/request.json` uses `"planner": "gemini"`. Request fields include `gemini_preset`, `gemini_models` (per-stage overrides, e.g. `{"writing": "gemini-2.5-pro"}`), `analysis_quality`, `review_passes`, `max_budget_usd`, `fallback_writer`, `forced_alignment` and `strict_review`. To run the original local pipeline instead, set `"planner": "local"` and start Ollama:

```bash
ollama serve
```

Then, in a second terminal with the environment activated:

```bash
python scripts/prepare_models.py
```

```bash
python -m shorts_pipeline doctor
```

Optional real-API checks (a handful of requests; free tier is fine) run only when you ask for them, locally or through **Actions → Gemini Live Check**:

```bash
RUN_GEMINI_LIVE=1 python -m pytest tests/test_gemini_live.py -v -s
```

Edit `examples/request.json` with a real local source path or HTTPS URL. Optional `source_window: [51, 73]` limits analysis to those original-source seconds. Writer/vision names and Ollama URL can be overridden in the request. `target_seconds` is a maximum, not an instruction to stretch speech. Real synthesized duration drives the edit; over-budget speech and insufficient footage fail with an actionable error. A bounded compression pass can rewrite overlong narration against its fixed shot budget; source quotes and selected shots remain unchanged and every attempt is retained for review. Output may be shorter than the chosen maximum. Use a fresh output folder for each run; completed results are preserved rather than overwritten.

For a reviewed edit, add `"plan": "work/storyboard.json"` to the request. A storyboard contains `beats` with `id`, `kind`, `text`, `delivery`, and `shots` (`start`, `end`, optional horizontal `focus_x`). Narration is synthesized with the selected voice. A `dialogue` beat keeps one contiguous source range and its audio. Reviewed narration files can instead be supplied as `audio_path` with relative per-word `words` timestamps. Imported word timings are validated against every written token and the measured audio. Audio paths are resolved relative to the storyboard file.

```json
{
  "beats": [
    {
      "id": "opening", "kind": "narration", "delivery": "dramatic",
      "text": "A stranger steps inside, but someone is already waiting.",
      "shots": [{"start": 2.0, "end": 8.0, "focus_x": 0.5}]
    },
    {
      "id": "exchange", "kind": "dialogue",
      "text": "Replace this with the verified words spoken in the source.",
      "shots": [{"start": 12.0, "end": 16.0}]
    }
  ]
}
```

The example demonstrates the schema; those words are not a claim about your footage. Automatic planning can now select short English source-speech passages from a fully retained ASR transcript. It copies recognized words and source times by ID, reserves their time, and does not invent quotes or speaker identities. Reviewed storyboards remain available for precise dialogue choices and other languages. The source analysis records confidence and rejected candidates; ASR still needs final playback review. Crop focus is explicit per shot, not an automatic guarantee of face tracking. Optional local `music` can be added in the request; source audio is reduced under narration and retained for dialogue.

## Models and timing (local pipeline and shared components)

- Gemini path: models are chosen per stage from preset candidate lists and checked against `models.list()` at runtime (see [architecture](docs/gemini-architecture.md)). Edit timing comes from measured Chatterbox audio. Caption word timings are Whisper estimates, optionally refined by torchaudio MMS forced alignment (`alignment=auto`), which is rejected if it disagrees with the ASR estimate.
- Writer: default `gemma4:12b`, drawing on the Top10 model selection. Its first saved-case draft and repair produced coherent narration covering the suit, broken rule and bounty. The smaller `qwen3:4b-instruct-2507-q4_K_M` remains a legacy option after repeated timing/quality failures. This is a case-specific comparison, not a broad writing benchmark. See the [validation record](docs/validation.md) for full pipeline coverage.
- Frame descriptions: explicit `qwen3-vl:2b-instruct-q4_K_M`, one midpoint image per indexed span. The floating `2b` alias selected a thinking model in a runner test, so the instruction variant is selected deliberately. Scene boundaries prefer PySceneDetect AdaptiveDetector, with a labeled FFmpeg fallback.
- Voice: pinned Chatterbox original/Turbo/Nano and the three existing owner-supplied references. Chatterbox's current upstream README describes it as retired research; the adapter is separate so it can be replaced. The pinned code still declares a mutable Perth dependency; inspect dependency resolution when upgrading.
- Source speech: multilingual faster-whisper `base` CPU int8, configurable through `SHORTS_SOURCE_WHISPER_MODEL`; automatic retained-speech selection currently admits English utterances only. Narration verification: faster-whisper CPU int8, default `base.en`, configurable through `SHORTS_WHISPER_MODEL`. Word timestamps are **ASR estimates**, not WhisperX forced alignment. Captions preserve script text and record timing provenance. The default burn-in progressively reveals and fades white words using the observed reference behavior; SRT remains plain text. Set `caption_style` to `static` in a CLI request to disable that animation.
- Final editing: FFmpeg hard cuts and focus crops; measured two-pass audio normalization targets −14 LUFS and −1.5 dBTP. This is our chosen output policy, not a uniform measured setting in the references.

New model-generated scripts and crops require editorial review. Evidence links and automated checks help identify issues; they do not establish factual accuracy or indistinguishable style.

## Tests and Graphify

```bash
python -m pytest -q
```

Gemini interfaces are tested offline with a scripted fake transport and the installed SDK's own request types; ordinary CI never calls the API.

Tests include actual synthetic A/V renders with audio checks, captions, duration/coverage validation, source and planner failures, and a complete supplied-audio CLI run. Synthetic tones exercise the machinery; they are not narrator-quality tests.

Graphify 0.9.102 was installed separately with video dependencies and a project Codex skill. Code extraction works offline; its video ingestion transcribes speech. Use it to connect the detailed reference observations with implementation decisions. Install the official package and project integration with:

```bash
uv tool install 'graphifyy[video]==0.9.102'
graphify install --project --platform codex
```

Keep original media and timestamped evidence: a graph is a retrieval aid and can omit details. Source video files remain in the [reference release](https://github.com/justsomeoneretarded/Recap-Short-form/releases/tag/references-v1), rather than in Git history.
