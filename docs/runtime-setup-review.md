# Runtime setup verification

This is the historical preflight inspection. Subsequent inference, render results and the current model defaults are recorded in [validation](validation.md); Gemma4 12B is the current writing baseline; the compact writers are evaluated separately with saved-case probes.

Read-only upstream inspection and HTTP metadata checks. No speech packages or model weights were installed by this review; no inference was run. Existing reference-transcription weights were not modified.

## Pinned Chatterbox package

Source: https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/pyproject.toml

The selected revision is `chatterbox-tts==0.1.7`, Python >=3.10. Actual declared dependencies include:

- NumPy >=1.24,<2 for Python <3.13; NumPy >=2 for Python >=3.13.
- `torch==2.6.0` and `torchaudio==2.6.0` for Python <3.14.
- `transformers==5.2.0`, `diffusers==0.29.0`, `librosa==0.11.0`, `safetensors==0.5.3`.
- `s3tokenizer`, `conformer==0.3.2`, `spacy-pkuseg`, `pykakasi==2.3.0`, `gradio==6.8.0`, `pyloudnorm`, `omegaconf`.
- `resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master`.

The Chatterbox commit is pinned, but its Perth Git dependency is still mutable. Do not describe the complete environment as immutably pinned. A recorded resolved environment / `pip freeze` is needed for production reproducibility. A constraints file cannot casually replace an upstream direct Git reference without resolver conflict; verify any rewritten dependency install separately.

Use Python 3.11 or 3.12 in a dedicated Shorts venv, separate from Graphify. A sensible install order for the existing chosen revision is:

```bash
python -m venv work/shorts-venv
work/shorts-venv/bin/python -m pip install --upgrade pip
work/shorts-venv/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
work/shorts-venv/bin/python -m pip install -e '.[voice,analysis,source]'
work/shorts-venv/bin/python -m pip check
```

Torchvision 0.21.0 matches Torch 2.6.0 and is used by Top10, although it is not listed directly in this Chatterbox pyproject. Installing the CPU build explicitly avoids accidentally pulling a large CUDA stack through transitive dependencies. The commands above are recommendations, not a completed installation test.

## Actual model files and download footprint

Public Hugging Face API metadata returned successfully without authentication. Both repos reported `gated:false` at inspection.

### Original

`tts.py` calls `hf_hub_download` for precisely these files from `ResembleAI/chatterbox`:

| File | Bytes |
|---|---:|
| t3_cfg.safetensors | 2,129,653,744 |
| s3gen.safetensors | 1,056,484,620 |
| ve.safetensors | 5,695,784 |
| conds.pt | 107,374 |
| tokenizer.json | 25,470 |

Total **3,191,966,992 bytes**, about 2.97 GiB, before Python packages and runtime memory. Resolved HF repository SHA observed: `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. Upstream `from_pretrained` does not pass an immutable `revision` argument.

Source: https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/tts.py

### Nano

`ChatterboxTurboTTS.from_local(..., nano=True)` uses `t3_nano_v1.safetensors`, `s3gen_meanflow.safetensors`, `ve.safetensors`, optional `conds.pt`, and GPT-2 tokenizer files from `ResembleAI/chatterbox-nano`.

- T3: 869,899,204 bytes.
- S3 meanflow: 1,064,875,036 bytes.
- Voice encoder: 5,695,784 bytes.
- Conditionals and tokenizer files: about 1.63 MB.
- Necessary runtime set: about **1.94 GB**.

Important: `from_pretrained(nano=True)` calls `snapshot_download` with broad allow patterns (`*.safetensors`, `*.json`, `*.txt`, `*.pt`, `*.model`). The current Nano repo also contains an unused `s3gen.safetensors` of 1,056,484,620 bytes. Therefore the stock loader downloads roughly **3.0 GB**, not just the smaller runtime set. Selective `hf_hub_download` plus `from_local` could avoid this extra GB, but that alternative must be implemented and tested explicitly. Do not silently assume the smaller footprint while calling stock `from_pretrained`.

Nano HF SHA observed: `71ccd1d0081b430592cea481f4307e764e07bc64`. Its stock loader reads `HF_TOKEN` if present but metadata is currently public; no token was needed for this inspection. The reference prompt must exceed five seconds; the supplied 12-second clips meet that source-code assertion.

Source: https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/tts_turbo.py

Metadata URLs checked:

- https://huggingface.co/api/models/ResembleAI/chatterbox?blobs=true
- https://huggingface.co/api/models/ResembleAI/chatterbox-nano?blobs=true

## Ollama availability and capacity

Registry manifest metadata returned successfully for both configured tags:

| Tag | Sum of manifest layers |
|---|---:|
| qwen3-vl:2b | 1,889,519,136 bytes |
| gemma4:12b | 8,021,618,699 bytes |

These checks establish tag availability and advertised file sizes, not model quality or inference success.

- https://registry.ollama.ai/v2/library/qwen3-vl/manifests/2b
- https://registry.ollama.ai/v2/library/gemma4/manifests/12b

The Top10 pinned Ollama `v0.40.14` release asset exists: HTTP HEAD returned 200 and 1,525,335,029 bytes at https://github.com/ollama/ollama/releases/download/v0.40.14/ollama-linux-amd64.tar.zst . This is the combined archive; Top10 excludes CUDA/Vulkan directories while extracting and deletes the archive before subsequent model downloads. Its configured checksum is `993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d`, which this review did not re-download and recompute.

Current workspace measurements: free disk 6,337,835,008 bytes; host-reported RAM 9.7 GiB but **cgroup memory.max is 8,589,934,592 bytes (8 GiB)**. The actual cgroup cap governs execution. The originally proposed8.02GB Gemma4 download alone exceeds current free disk, before adding runtime/dependencies. Do not attempt a full default-model pipeline here and call a resource failure a model-quality issue.

On a suitable GitHub runner, sequence stages: index/describe, unload vision; write plan, unload writer; synthesize; render. Consider deleting no-longer-needed Ollama model blobs between stages on constrained disk, rather than retaining vision + writer + TTS weights simultaneously. Merely setting `keep_alive:0` releases RAM but does not free model disk. Preserve user-managed/shared caches; cleanup should operate only on owned per-run model storage.

## Feasible validation recommendation

1. Run the model-free supplied-audio/word-timing render tests now; label these as mechanical tests, not generated narration tests.
2. An actual short Nano speech check could fit this workspace **only after** an isolated CPU dependency install and explicit disk preflight. Prefer the selective Nano runtime file set if implemented; otherwise budget ~3 GB of model downloads plus the venv and pip cache. Use `--no-cache-dir` or a controlled cache to avoid duplicate wheel storage, and preserve the original reference files.
3. Generate one short complete sentence (~20–30 words) with the Warguy reference. Measure the real waveform, check non-silence/finite samples/word completion using existing faster-whisper, and record elapsed CPU time and model/code revisions. Do not infer voice likeness from those numerical checks; subjective audition remains separate.
4. An original-Chatterbox validation requires its larger 3.19 GB weight set and may be tight alongside dependencies and this workspace's media/models. Use a larger runner if capacity checks fail.
5. Run the complete selected writer + Qwen3-VL + original-Chatterbox production path on a runner that actually fits it, with `pip check` and captured status/manifests. A small-model substitution tests API plumbing, not the configured model's output quality.

No access credentials are required just to install these public packages and weights. Real download/inference can still fail and must be reported as such.
