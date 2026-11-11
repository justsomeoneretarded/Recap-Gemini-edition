# Upstream technology comparison for Recap Shorts

Reviewed 5 November 2026 (Africa/Nairobi). These are ten relevant candidates, not a claim that they are the world's ten best repositories. Recommendations are based on inspected upstream docs and repository metadata, not end-to-end performance tests. Library code licenses do not automatically license model weights, source footage, music, or fonts.

## Recommended initial composition

Use direct-file/GitHub-release inputs plus optional yt-dlp, FFmpeg, PySceneDetect, faster-whisper, configurable Ollama, pinned Chatterbox with the existing three references, and optional OpenCLIP ranking. Add WhisperX alignment where measured caption timing warrants its cost. Keep MoviePy and Remotion outside the first production dependency set.

The strongest specific additions to the Top10 approach are PySceneDetect's adaptive cut detection for fast action and a replaceable forced-alignment stage for word-accurate captions. Neither is a substitute for source-grounded story selection. For covers, prefer clear source frames, crop/readability scoring, and deterministic text composition before adding generative-image or browser-render stacks.

## Comparison

| Repository | Role | CPU suitability | Recommendation |
|---|---|---|---|
| [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) | Source-video acquisition and optional subtitles | Good; network/decode work, no GPU needed. | Use as optional downloader, with direct-file and GitHub-release download support as reliable alternatives. YouTube access depends on the host/network and current extractor requirements; do not treat successful installation as successful download. |
| [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) | Decode, trimming, crop/scale, concat, captions, audio mixing, loudness, final encode, cover-frame extraction | Good; bound thread count, use appropriate preset and avoid unnecessary repeated full encodes. | Core renderer and media-validation engine. Prefer a timeline expressed as concrete source ranges and filters, followed by ffprobe validation. |
| [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Shot boundaries and scene segmentation | Good; CPU video decode with configurable downscaling. | Use AdaptiveDetector for action footage and verify boundary candidates. Keep minimum shot length and allow explicit user ranges. It identifies visual transitions, not story beats. |
| [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Transcription, speech verification, approximate word timing | Strong candidate: CPU INT8 supported; small/base models practical. | Core transcription/verification component. Use exact authored script for captions where possible and validate alignment rather than copying ASR mistakes into final text. |
| [m-bain/whisperX](https://github.com/m-bain/whisperX) | Forced word alignment and optional diarization | Supported with --compute_type int8 --device cpu, but adds models/dependencies and processing cost. | Optional quality tier for tight word-to-caption timing; omit diarization for a single narrator unless needed. Not required in first lightweight path. |
| [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | Text-to-frame retrieval and narration-to-shot candidate ranking | Feasible in small batches with smaller encoders and sparse scene representatives; large SigLIP2 models can be costly. | Use optional semantic ranking with Top10's pinned, compatible inference setup. Preserve continuity and source time constraints; similarity is not proof that an event occurred. |
| [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | Reference-conditioned narration using Warguy, Kit Laser, Female samples | Nano explicitly targets CPU (110M); Turbo 350M; original and multilingual variants heavier. Actual voice quality and speed need auditions. | Retain Top10's known pinned revision and all three reference samples, with configurable model choice. Keep a replaceable TTS interface because maintenance is uncertain. Do not silently upgrade. |
| [ollama/ollama](https://github.com/ollama/ollama) | Local model serving for evidence-grounded scripts and structured beat plans | Supported; speed and memory depend on model size, quantization and context. | Use configurable local endpoint and structured JSON responses with schema validation. Select a practical model using Top10 review and actual runtime limits; never imply the server itself establishes writing quality. |
| [Zulko/moviepy](https://github.com/Zulko/moviepy) | Python compositing, effects, title insertion and prototypes | Supported; upstream explicitly notes heavier array import/export and slower operation than direct FFmpeg. | Do not require it for the production core. It is useful for rapid effects prototypes, but direct FFmpeg covers current needs with lower overhead. |
| [remotion-dev/remotion](https://github.com/remotion-dev/remotion) | React-based animated titles, graphics and cover/template composition | Browser rendering can run on CPU, but adds Node/browser dependencies and resource cost. | Optional future template/graphics layer, not first pipeline dependency. FFmpeg and image composition are enough for current scene-driven recap style. |

## yt-dlp/yt-dlp

- **License:** Unlicense for repo/PyPI wheel. Bundled release executables may include GPLv3+ components.
- **Evidence and limits:** Current README requires Python 3.11+, recommends FFmpeg/ffprobe and yt-dlp-ejs plus a supported JavaScript runtime for full YouTube support.
- **Maintenance snapshot:** default branch `master`; latest default-branch commit 2026-11-04T07:16:26Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`412e7f0a4b0588fe884a02b344ac66a06538475b`](https://github.com/yt-dlp/yt-dlp/commit/412e7f0a4b0588fe884a02b344ac66a06538475b).
- **Primary documentation:** [README at inspected revision](https://github.com/yt-dlp/yt-dlp/blob/412e7f0a4b0588fe884a02b344ac66a06538475b/README.md).

## FFmpeg/FFmpeg

- **License:** Mainly LGPL-2.1-or-later. Optional GPL components and linked libraries change build license; libx264/libx265 require GPL-enabled builds. Inspect the installed binary configuration.
- **Evidence and limits:** Upstream GitHub repository is a mirror; README directs contributions to Forgejo/mailing list, not GitHub pull requests.
- **Maintenance snapshot:** default branch `master`; latest default-branch commit 2026-11-05T04:43:58Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`0893feba652894f836cb76617936da52570d20a5`](https://github.com/FFmpeg/FFmpeg/commit/0893feba652894f836cb76617936da52570d20a5).
- **Primary documentation:** [README at inspected revision](https://github.com/FFmpeg/FFmpeg/blob/0893feba652894f836cb76617936da52570d20a5/README.md).

## Breakthrough/PySceneDetect

- **License:** BSD-3-Clause; third-party dependencies have separate terms.
- **Evidence and limits:** README explicitly says the two-pass AdaptiveDetector handles fast camera movement better than ContentDetector. ThresholdDetector handles fade-out/fade-in.
- **Maintenance snapshot:** default branch `main`; latest default-branch commit 2026-11-02T01:18:02Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`c5ce29c73751e77139e838bd9ce9ab66a8e71a23`](https://github.com/Breakthrough/PySceneDetect/commit/c5ce29c73751e77139e838bd9ce9ab66a8e71a23).
- **Primary documentation:** [README at inspected revision](https://github.com/Breakthrough/PySceneDetect/blob/c5ce29c73751e77139e838bd9ce9ab66a8e71a23/README.md).

## SYSTRAN/faster-whisper

- **License:** MIT repository code; model weights/dependencies must be checked separately.
- **Evidence and limits:** README provides CPU INT8 and word_timestamps examples. Published small-model CPU benchmark is upstream evidence, not a benchmark on our 3-CPU workspace.
- **Maintenance snapshot:** default branch `master`; latest default-branch commit 2026-11-02T11:19:10Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`2a3a857737d17d974f30d818ebec69ce11929e5c`](https://github.com/SYSTRAN/faster-whisper/commit/2a3a857737d17d974f30d818ebec69ce11929e5c).
- **Primary documentation:** [README at inspected revision](https://github.com/SYSTRAN/faster-whisper/blob/2a3a857737d17d974f30d818ebec69ce11929e5c/README.md).

## m-bain/whisperX

- **License:** BSD-2-Clause repository code. Diarization/alignment checkpoints carry their own terms; README identifies pyannote speaker-diarization-community-1 as CC-BY-4.0.
- **Evidence and limits:** Provides wav2vec2-based alignment after ASR. Some tokens such as numbers/currency may not align; handle missing timings explicitly.
- **Maintenance snapshot:** default branch `main`; latest default-branch commit 2026-10-22T14:53:43Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`5d3678b8535698a2826a5db9eb3e4105aef85dd9`](https://github.com/m-bain/whisperX/commit/5d3678b8535698a2826a5db9eb3e4105aef85dd9).
- **Primary documentation:** [README at inspected revision](https://github.com/m-bain/whisperX/blob/5d3678b8535698a2826a5db9eb3e4105aef85dd9/README.md).

## mlfoundations/open_clip

- **License:** LICENSE text is MIT even though GitHub metadata says NOASSERTION/Other. Model checkpoints and datasets have separate licenses.
- **Evidence and limits:** Current README supports SigLIP/SigLIP2 families and requires PyTorch>=2.6. Avoid blindly pulling latest training API changes into an inference-only application.
- **Maintenance snapshot:** default branch `main`; latest default-branch commit 2026-10-29T21:28:40Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`4308ec82892b2d7d27aa4eb309973b5bd2c81a73`](https://github.com/mlfoundations/open_clip/commit/4308ec82892b2d7d27aa4eb309973b5bd2c81a73).
- **Primary documentation:** [README at inspected revision](https://github.com/mlfoundations/open_clip/blob/4308ec82892b2d7d27aa4eb309973b5bd2c81a73/README.md).

## resemble-ai/chatterbox

- **License:** MIT repository code. Check chosen checkpoint terms separately; preserve built-in watermark behavior.
- **Evidence and limits:** Current README banner says 'As of August 2026, Chatterbox is retired research and is no longer maintained.' GitHub archived=false and recent push remain visible; those metadata signals do not override the explicit maintenance statement. Nano's claimed 3x realtime on 8 CPU cores is an upstream claim, not a guarantee here.
- **Maintenance snapshot:** default branch `master`; latest default-branch commit 2026-11-04T18:48:10Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`ab55923c27757c90cfcc407baa57dc25a7d25217`](https://github.com/resemble-ai/chatterbox/commit/ab55923c27757c90cfcc407baa57dc25a7d25217).
- **Primary documentation:** [README at inspected revision](https://github.com/resemble-ai/chatterbox/blob/ab55923c27757c90cfcc407baa57dc25a7d25217/README.md).

## ollama/ollama

- **License:** MIT server/client code. Each served model has its own license; Ollama's license does not cover Gemma weights.
- **Evidence and limits:** Official FAQ describes CPU, GPU and mixed loading, local-only mode OLLAMA_NO_CLOUD=1, and OLLAMA_MODELS for custom storage.
- **Maintenance snapshot:** default branch `main`; latest default-branch commit 2026-11-05T01:31:38Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`a6d33251620eef7629c73b09e672f2df243eb363`](https://github.com/ollama/ollama/commit/a6d33251620eef7629c73b09e672f2df243eb363).
- **Primary documentation:** [README at inspected revision](https://github.com/ollama/ollama/blob/a6d33251620eef7629c73b09e672f2df243eb363/README.md).

## Zulko/moviepy

- **License:** MIT (GitHub license metadata).
- **Evidence and limits:** README notes major v2 breaking changes; v1 is unmaintained. Maintainers request help and acknowledge variable bandwidth.
- **Maintenance snapshot:** default branch `master`; latest default-branch commit 2026-08-26T06:17:07Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`211e4b15f6ce4f34a6a9efbfff40590e43a68f77`](https://github.com/Zulko/moviepy/commit/211e4b15f6ce4f34a6a9efbfff40590e43a68f77).
- **Primary documentation:** [README at inspected revision](https://github.com/Zulko/moviepy/blob/211e4b15f6ce4f34a6a9efbfff40590e43a68f77/README.md).

## remotion-dev/remotion

- **License:** Custom Remotion license. Current text permits free use by individuals, for-profit organizations with up to 3 employees, nonprofits, or evaluation; other organizations require a company license. It is not a standard permissive OSS license.
- **Evidence and limits:** License text warns Remotion 5.0 terms will change; recheck eligibility and version before later adoption.
- **Maintenance snapshot:** default branch `main`; latest default-branch commit 2026-11-04T16:25:34Z; repository archived=false. A recent commit is activity evidence, not a support guarantee.
- **Inspected revision:** [`682cad29c569d7c36c2445e27ec07cb433f06b32`](https://github.com/remotion-dev/remotion/commit/682cad29c569d7c36c2445e27ec07cb433f06b32).
- **Primary documentation:** [README at inspected revision](https://github.com/remotion-dev/remotion/blob/682cad29c569d7c36c2445e27ec07cb433f06b32/README.md).

## License sources and checkpoints

- [FFmpeg detailed build-dependent license](https://github.com/FFmpeg/FFmpeg/blob/0893feba652894f836cb76617936da52570d20a5/LICENSE.md)
- [OpenCLIP MIT license text](https://github.com/mlfoundations/open_clip/blob/4308ec82892b2d7d27aa4eb309973b5bd2c81a73/LICENSE)
- [WhisperX BSD-2-Clause license](https://github.com/m-bain/whisperX/blob/5d3678b8535698a2826a5db9eb3e4105aef85dd9/LICENSE)
- [Chatterbox MIT license](https://github.com/resemble-ai/chatterbox/blob/ab55923c27757c90cfcc407baa57dc25a7d25217/LICENSE)
- [Remotion custom license](https://github.com/remotion-dev/remotion/blob/682cad29c569d7c36c2445e27ec07cb433f06b32/LICENSE.md)
- [Ollama CPU/local-only/storage FAQ](https://github.com/ollama/ollama/blob/a6d33251620eef7629c73b09e672f2df243eb363/docs/faq.mdx)

GitHub repository API metadata supplied the other SPDX identifiers, checked alongside each README. No upstream packages were installed or benchmarked for this review. Exact source SHAs above document what was inspected; they are not automatic recommendations to upgrade to those heads.
