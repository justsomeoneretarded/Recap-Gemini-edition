"""Optional acoustic forced alignment of known narration text (torchaudio MMS_FA).

The script text is known exactly, so forced alignment can locate each word more
reliably than matching an independent ASR transcript. It runs only when
requested and torchaudio (already installed with the voice extra) can load the
MMS_FA bundle; otherwise the ASR-estimated timings remain, labelled as such.
A forced result that disagrees grossly with the ASR estimate is rejected.
"""
from __future__ import annotations

import re
import statistics

PROVENANCE = "mms-forced-alignment"
_STATE: dict = {}


def _romanize(word: str) -> str:
    from .voice import _lexical
    pieces = _lexical(word)
    return "".join(re.sub(r"[^a-z']", "", p) for p in pieces)


def available() -> bool:
    try:
        import torchaudio  # noqa: F401
        return hasattr(__import__("torchaudio").pipelines, "MMS_FA")
    except ImportError:
        return False


def _load():
    if "bundle" not in _STATE:
        import torch
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        _STATE.update(torch=torch, torchaudio=torchaudio, bundle=bundle,
                      model=bundle.get_model(with_star=False), tokenizer=bundle.get_tokenizer(),
                      aligner=bundle.get_aligner())
    return _STATE


def word_spans_from_tokens(spans: list[list], frames: int, samples: int, sample_rate: int) -> list[tuple[float, float]]:
    ratio = samples / frames / sample_rate
    return [(group[0].start * ratio, group[-1].end * ratio) for group in spans]


def check_against_estimate(forced: list[dict], estimate: list[dict], tolerance: float = 0.6) -> bool:
    if len(forced) != len(estimate):
        return False
    deltas = [abs(f["start"] - e["start"]) for f, e in zip(forced, estimate)]
    ordered = all(b["start"] >= a["start"] - 1e-6 for a, b in zip(forced, forced[1:]))
    return ordered and statistics.median(deltas) <= tolerance


def refine_words(audio_path: str, words: list[dict], duration: float) -> tuple[list[dict], str]:
    """Return (words, provenance); unchanged ASR words when alignment is unsuitable."""
    transcript = [_romanize(w["word"]) for w in words]
    if not words or any(not t for t in transcript):
        return words, "asr-estimated-script-matched"
    state = _load()
    torch, torchaudio, bundle = state["torch"], state["torchaudio"], state["bundle"]
    waveform, rate = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform.mean(0, keepdim=True), rate, bundle.sample_rate)
    with torch.inference_mode():
        emission, _ = state["model"](waveform)
    spans = state["aligner"](emission[0], state["tokenizer"](transcript))
    times = word_spans_from_tokens(spans, emission.size(1), waveform.size(1), bundle.sample_rate)
    forced = [{"word": w["word"], "start": round(a, 3), "end": round(min(b, duration), 3)} for w, (a, b) in zip(words, times)]
    if not check_against_estimate(forced, words):
        return words, "asr-estimated-script-matched (forced alignment rejected: disagreed with ASR estimate)"
    return forced, PROVENANCE
