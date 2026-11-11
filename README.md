# Recap Shorts

A standalone movie recap Shorts pipeline with its own **Generate Shorts** workflow. Choose footage, a story brief, a narrator, a maximum duration, and square or vertical output. This repository does not trigger or depend on long-form generation.

The initial style profile comes from eight owner-supplied references: action before backstory, causal story beats, compact white italic captions, meaningful object/face shots, selected movie dialogue, and story-dependent endings. All supplied exports are 1080×1080, roughly 136–180 seconds. Square is the default; vertical is an explicit option. This is an initial implementation, not a verified recreation of the creator's production process.

Read the [reference analysis](docs/reference-analysis.md), [ten-repository comparison](docs/technology-comparison.md), and [validation record](docs/validation.md). The [complete automatic workflow passed](https://github.com/justsomeoneretarded/Recap-Short-form/actions/runs/42165849180), producing a [14-second Warguy engineering preview](docs/validation-samples/automatic-warguy-pilot.mp4) at1080square/60fps. This test reused verified source analysis and generated a fresh script, voice and edit. The reference already has captions/narration, which overlap the generated content; the sample is not channel-ready.

## Generate a Short on GitHub

Open **Actions → Generate Shorts → Run workflow**. Enter:

| Field | What to provide |
|---|---|
| Title | Your Short's title |
| Source URL | A direct HTTPS URL to source footage, including a GitHub Release attachment URL |
| Brief | The scene/story angle and editing instructions |
| Story notes | Verified names, context and events the script may use |
| Narrator | Warguy, Kit Laser, or Female |
| Maximum seconds | 15–180; default 150 |
| Aspect | Square or vertical |
| Speech model | Original, Turbo or Nano |
| Writer | Gemma4 12B by default; Qwen3 4B is a smaller legacy option |
| Source window | Optional start,end in seconds to analyze a relevant excerpt |

Use clean movie/scene footage for production. The eight reference Shorts already contain narration and burned captions; they are style references and engineering test material. A movie title alone cannot supply movie footage. Use a relevant source excerpt containing the actions you want narrated. Automatic planning describes every indexed span using a representative midpoint frame; it does not continuously understand all motion or speech. The default 120-span cap fails explicitly on larger footage. For full films, select an excerpt using the Source window field, prepare an excerpt file, deliberately raise the CLI cap, or provide a reviewed storyboard. With a source window, storyboard times are relative to that excerpt; provenance records its offset in the original. YouTube downloading is supported through yt-dlp but can be blocked by the runner/network; direct media is the reliable alternative tested here.

After success, download **Artifacts → shorts-N**. It includes `final.mp4`, script and description, SRT/ASS captions, storyboard, measured timeline, three cover candidates, contact sheet, source provenance, caption-timing provenance, quality report and execution status. Cover candidates are sampled frames for selection, not evidence of an optimized thumbnail system. There is no automatic YouTube upload.

The workflow starts local models on the Actions runner; no paid inference API or TMDB key is required. The first run downloads substantial models and can take a long time on CPU. Account Actions usage/storage limits apply. The default writer Gemma4 12B is about 8 GB of weights; the optional legacy Qwen3 4B is about 2.5 GB. Qwen3-VL 2B is about 1.9 GB and Chatterbox weights about 3 GB. For the default Gemma chain, prefer at least 16 GB RAM and 25 GB free disk; the smaller Qwen chain needs about 18 GB free disk. Preflight rejects insufficient capacity before downloading models. These sizes establish feasibility, not editorial quality. The smaller local test workspace could not run the complete configured model chain; see validation.

## Local setup

Use Python 3.11 or 3.12, FFmpeg/FFprobe with libass, and DejaVu fonts. Keep this environment separate from Graphify's environment because their dependency versions differ.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[voice,analysis,source,test]'
```

Start Ollama separately, then prepare the configured writer and vision model:

```bash
ollama serve
# In a second terminal, with the environment activated:
python scripts/prepare_models.py
python -m shorts_pipeline doctor
python -m shorts_pipeline generate --request examples/request.json --output output/shorts
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

## Models and timing

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

Tests include actual synthetic A/V renders with audio checks, captions, duration/coverage validation, source and planner failures, and a complete supplied-audio CLI run. Synthetic tones exercise the machinery; they are not narrator-quality tests.

Graphify 0.9.102 was installed separately with video dependencies and a project Codex skill. Code extraction works offline; its video ingestion transcribes speech. Use it to connect the detailed reference observations with implementation decisions. Install the official package and project integration with:

```bash
uv tool install 'graphifyy[video]==0.9.102'
graphify install --project --platform codex
```

Keep original media and timestamped evidence: a graph is a retrieval aid and can omit details. Source video files remain in the [reference release](https://github.com/justsomeoneretarded/Recap-Short-form/releases/tag/references-v1), rather than in Git history.
