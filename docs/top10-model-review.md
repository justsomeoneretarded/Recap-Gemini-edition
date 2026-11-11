# Top10-Movies model and implementation review

Inspected local checkout `c6f82a621b9de57a2c180a74cd9e37e17f2a6a92` of `justsomeoneretarded/Top10-Movies`. This is source inspection, not a model-quality benchmark or production-run validation. Paths/line numbers below refer to this revision.

## Model choices actually configured

| Stage | Configured model/tool | Evidence and practical implications |
| --- | --- | --- |
| Script | `gemma4:12b` through Ollama | `movie_pipeline/script.py:336–340,414–425`: schema-constrained `/api/chat`, `think:false`, default 8192 context and 2300 output tokens, model override `SCRIPT_MODEL`; unload at line 941. `.github/workflows/generate-video.yml:485,498–508` budgets a roughly 8 GB model within a 16 GB runner, one loaded model/request, q8 KV cache. Do not assume a 3-CPU workspace has enough RAM or acceptable generation speed; preflight actual model pull and a small structured response. |
| Speech | Original Chatterbox, optional Turbo/Nano | `movie_pipeline/voice.py:25–50,168–192`; all CPU and selected reference prepared once. Original normal settings exaggeration .50, CFG .50, temperature .80; 45-word generation limit. Voice emotion and two-take selection are configurable. Keep original as compatibility default; audition actual output before claiming voice likeness. |
| Narrators | Warguy, Kit Laser, Female | Bundled `assets/voices/warguy.ogg`, `kit-laser.ogg`, `female.ogg`. `README.md:10–21` describes owner-supplied, approved 12-second mono references. No vendor voice IDs needed. Parent already verified durations. |
| Transcription / speech QA | faster-whisper `base.en`, `small.en` fallback | `movie_pipeline/cli.py:426–430,1083–1084`; CPU/int8, up to four threads. Useful reusable pattern: compare generated speech against intended script and retry incomplete takes. For reference analysis, parent is using `small.en` with explicit timestamps. |
| Visual-text matching | SigLIP 2 + CLIP | `movie_pipeline/shots.py:355–390`; preferred OpenCLIP model `hf-hub:timm/ViT-B-16-SigLIP2`, optional Transformers model `google/siglip2-base-patch16-224`, fallback `openai/clip-vit-base-patch32`. CLIP also retained for card/character heuristics whose thresholds were tuned to it. Similarity is a ranking signal, not proof that a frame depicts a specific story action. |
| Scene descriptions | `qwen3-vl:2b` through Ollama | `movie_pipeline/scenes.py:24–31,70–78,94–108`. Produces <=14-word descriptions, 448x252 JPEG input, short generation. **Defaults sample only six representative/impact frames per trailer within a 420-second budget**. This is not full-shot or full-frame understanding. |
| Text embeddings | `all-minilm` through Ollama | `movie_pipeline/scenes.py:25,62–67`. Embeds descriptions and narration for retrieval. Optional when SigLIP works. |
| Dialogue separation | Demucs `htdemucs` | `movie_pipeline/dialogue.py:1–7,27–36`; separates only selected short dialogue windows, falls back to original audio on failure. Optional enhancement; separate narration must not be mixed with unrelated reference narration. |
| Music | ACE-Step 1.5 `acestep-v15-turbo` | `scripts/generate_music.py:30–39`: CPU, instrumental, eight inference steps; failure intentionally falls back to bundled music. Avoid making generated music a first-production dependency. |
| Thumbnail cutouts | rembg CPU | `pyproject.toml:18` pins `rembg[cpu]==2.0.67`; first version can use selected source frames and face-aware crops instead of adding image generation. |

## Dependency/runtime pins worth retaining

- `pyproject.toml:9–20`: Python >=3.11,<3.13; NumPy >=1.26,<2; OpenCV >=4.10,<4.12; faster-whisper >=1.2,<2; HF Hub >=0.34,<2. These constraints differ from the separate Graphify installation (NumPy 2 / HF Hub 2), so **use a dedicated Shorts virtual environment**.
- Chatterbox is pinned to Git revision `5de7a54aa4e5e2baadb0182dde554908b48b85c2` (`pyproject.toml:19`). Production workflow installs CPU `torch==2.6.0`, `torchaudio==2.6.0`, `torchvision==0.21.0`, then `open_clip_torch==2.32.0` and `timm==1.0.15` without replacing Chatterbox's Transformers stack (`generate-video.yml:580–590`). Use a constraints file and `pip check`; unpinned mixed stacks can silently break.
- Ollama installer is pinned to version `0.40.14`, SHA-256 `993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d`, with Linux-amd64 CPU archive extraction (`scripts/install_ollama.py:23–36`). Verify remote release availability/checksum before copying this assumption into a new workflow. The inspected code alone is not runtime proof.
- Script model tags, CLIP/SigLIP model weights and Qwen weights are **not immutable revision-pinned** in this checkout. Record resolved model identifiers/checksums in run manifests for reproducibility.

## Reusable code ideas and changes needed for Shorts

1. Keep bounded phrase synthesis, pronunciation overrides, narration verification, generated-audio duration-driven edit planning, and resumable stage caches. Use one voice/model/reference throughout a short. Reuse tests that verify incomplete speech and frame/audio duration mismatches.
2. Keep deterministic FFmpeg assembly and output validation. `render.py:223–230` budgets whole frames from actual speech duration; `render.py:543` repairs audio/video duration alignment. Adapt aspect ratio and timing to observed references rather than adopting the long-form layout.
3. Use actual word timestamps for short captions. Existing `subtitles.py:22–34` spreads words across each cue proportionally by character length before script alignment; it is **approximate**, not precise forced alignment. Long-form limits of 84 characters/seven seconds (`subtitles.py:13–14`) need a Shorts-specific caption layout.
4. Do not carry over the ranking intro, subscribe break, title announcements, multi-film structure or long-form 4.5–10-second shot assumptions. Build a separate one-story outline: evidence-backed opening, setup, escalation, reveal and intentional stopping point, based on parent analysis.
5. A source film/scene plus transcript or supplied plot notes should be first-class inputs. TMDB overview + trailer is insufficient evidence for a detailed full-story recap. Keep script claims linked to source timestamps/notes so source selection does not invent missing actions.
6. Visual selection must enforce sufficient fresh footage and coherent ordering. Top10 intentionally treats relevance softly (`source_pool.py:34–44`, `README.md:97`); for scene recaps a semantically wrong shot can contradict narration. Store selected source interval and matching rationale for every narration beat.

## Source acquisition and workflow

`research.py:18–20` requires `TMDB_READ_TOKEN`. Title/year/content-type resolution is explicit (`research.py:77–149`), with official YouTube/Vimeo trailer metadata and official scene clips preferred. `footage.py:160–178,212–245` uses yt-dlp, validates identity and has bounded attempts. `footage.py:181–190` can fall back to stills. For Shorts, make an original video path or direct downloadable media URL a supported route so a YouTube/TMDB failure does not block supplied footage. Report any stills fallback as a degraded mode rather than implying scene footage was used.

The workflow is manually dispatched and reads user fields through environment variables, has read-only repository permissions and serialized production (`generate-video.yml:4–44,115–120`). Keep that pattern for a distinct `Generate Shorts` workflow. Its long-form production job has a 350-minute timeout (`:470–475`); this is an upper bound, **not evidence of typical runtime**. Split optional resource-heavy work from the core render, checkpoint manifests and upload logs/report plus output artifacts. Never interpolate user text directly into shell script source.

## Licenses actually evidenced in this checkout

- No root `LICENSE`, `COPYING` or `NOTICE` file and no `license` field in `pyproject.toml` were found. The user's ownership of both repositories supports adapting their code within this task, but do not invent an MIT/Apache license for Top10-Movies or add one without a basis.
- `assets/music/LICENSE.md` explicitly identifies **Creative Commons Attribution 4.0 International** for Scott Buckley's Titan, Celestial and Aurora; preserve track-specific attribution and provenance if reused. These files are excluded from any repository code license by the notice.
- README calls ACE-Step 1.5 MIT, but that is a README assertion, not an upstream LICENSE audit. Upstream licenses/model terms for Gemma, Chatterbox, Qwen, SigLIP, CLIP, Demucs, rembg, faster-whisper and Ollama were not fetched in this narrow inspection. Parent's broader repository comparison should verify those separately before labeling them.

## Recommendation

Start with a standalone CLI + manually dispatched GitHub Actions workflow. Require supplied source footage/URL and story direction, support optional explicit script for deterministic tests, reuse the three voice references, use original Chatterbox by default, faster-whisper word timing, source-shot indexing with optional SigLIP2, and FFmpeg render/captions/audio mix. Make Gemma4/Ollama a selectable, preflighted writer rather than a silent assumption. Keep optional music generation, Demucs, vision captioning and complex thumbnail generation out of the critical path until core production is verified. A synthetic or supplied-script render can validate mechanics, but label it clearly and separately from a fully generated script+voice production run.
