"""Measured source speech for optional original-audio beats.

ASR timings are estimates, not forced alignment or speaker identification.
The complete transcript is retained even when no utterance passes admission.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path

from .sources import _video_info, _write_json, file_digest

SOURCE_SPEECH_VERSION = 2
TIMING_PROVENANCE = "asr-estimated-source-speech"


def _number(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _transcribe(source: Path, model_name: str, threads: int) -> dict:
    # Import and allocate only on cache misses; no persistent model competes
    # with the visual model for CPU/RAM after this function returns.
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=threads, num_workers=1)
    try:
        stream, info = model.transcribe(
            str(source), task="transcribe", beam_size=3,
            word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = []
        for segment in stream:
            segments.append({
                "id": segment.id, "start": _number(segment.start),
                "end": _number(segment.end), "text": segment.text,
                "avg_logprob": _number(segment.avg_logprob),
                "no_speech_prob": _number(segment.no_speech_prob),
                "compression_ratio": _number(segment.compression_ratio),
                "words": [{"word": word.word, "start": _number(word.start),
                           "end": _number(word.end),
                           "probability": _number(word.probability)}
                          for word in (segment.words or [])],
            })
        return {"language": info.language,
                "language_probability": _number(info.language_probability),
                "segments": segments}
    finally:
        del model
        gc.collect()


def _candidate(segment: dict, duration: float) -> tuple[dict | None, str | None]:
    start, end = _number(segment.get("start")), _number(segment.get("end"))
    if start is None or end is None or start < 0 or end <= start or end > duration:
        return None, "invalid_segment_bounds"
    if not .4 <= end - start <= 8:
        return None, "duration_outside_0.4_to_8_seconds"
    words = segment.get("words") or []
    if not 2 <= len(words) <= 28:
        return None, "word_count_outside_2_to_28"
    previous = start
    clean, probabilities = [], []
    for word in words:
        first, last = _number(word.get("start")), _number(word.get("end"))
        text = str(word.get("word", "")).strip()
        # Each ASR word must remain a single caption token. Reject rather than
        # invent timings when the recognizer returns a multiword token.
        if not text or len(text.split()) != 1:
            return None, "invalid_word_text"
        if (first is None or last is None or first < start or last <= first
                or last > end or first < previous - .04):
            return None, "invalid_word_bounds"
        probability = _number(word.get("probability"))
        if probability is None or not 0 <= probability <= 1:
            return None, "invalid_word_confidence"
        previous = last
        probabilities.append(probability)
        clean.append({"word": text, "start": first - start, "end": last - start,
                      "probability": probability})
    confidence = sum(probabilities) / len(probabilities)
    if confidence <= .75:
        return None, "low_word_confidence"
    no_speech = _number(segment.get("no_speech_prob"))
    if no_speech is None or not 0 <= no_speech < .35:
        return None, "high_or_missing_no_speech_probability"
    return {"start": start, "end": end,
            "text": " ".join(word["word"] for word in clean),
            "words": clean, "confidence": confidence,
            "no_speech_prob": no_speech,
            "source_segment_id": segment.get("id"),
            "timing_provenance": TIMING_PROVENANCE}, None


def analyze_source_speech(source: Path, workdir: Path) -> dict:
    """Transcribe full supplied footage and admit English source utterances.

    All times refer to the supplied source (including when it is an excerpt).
    A source with too many admissible utterances requires a narrower excerpt;
    its complete evidence is saved before the explicit admission error.
    """
    source, workdir = Path(source), Path(workdir)
    model_name = os.environ.get("SHORTS_SOURCE_WHISPER_MODEL", "base").strip()
    if not model_name:
        raise ValueError("SHORTS_SOURCE_WHISPER_MODEL must name a Whisper model.")
    threads = int(os.environ.get("SHORTS_SOURCE_WHISPER_THREADS", "2"))
    maximum = int(os.environ.get("SHORTS_MAX_SOURCE_SPEECH_CANDIDATES", "100"))
    if threads < 1 or maximum < 1:
        raise ValueError("Source speech CPU threads and candidate limit must be positive.")
    try:
        asr_version = version("faster-whisper")
    except PackageNotFoundError:
        asr_version = "unavailable"
    digest = file_digest(source)
    key = hashlib.sha256(json.dumps(
        {"source": digest, "model": model_name, "version": SOURCE_SPEECH_VERSION,
         "asr_library_version": asr_version},
        sort_keys=True).encode()).hexdigest()
    path = workdir / "source-speech" / f"{key}.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached.get("cache_key") == key and cached.get("complete") is True
                and cached.get("source_sha256") == digest):
            if len(cached["candidates"]) > maximum:
                raise ValueError(f"Source speech has {len(cached['candidates'])} candidates; "
                                 f"limit is {maximum}. Supply a narrower source excerpt. "
                                 f"Full transcript: {path}")
            return cached
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        pass
    info = _video_info(source)
    duration = info["duration"]
    transcript = (_transcribe(source, model_name, threads) if info["has_audio"]
                  else {"language": None, "language_probability": None, "segments": []})
    candidates, rejections = [], []
    language = transcript.get("language")
    reason = ("source_has_no_audio" if not info["has_audio"] else
              "automatic_source_speech_requires_english" if language != "en" else None)
    for index, segment in enumerate(transcript["segments"]):
        candidate, rejected = (None, reason) if reason else _candidate(segment, duration)
        if candidate is None:
            rejections.append({"segment_index": index, "source_segment_id": segment.get("id"),
                               "reason": rejected})
        else:
            candidate["id"] = f"D{len(candidates) + 1:04d}"
            candidates.append(candidate)
    result = {"version": SOURCE_SPEECH_VERSION, "complete": True,
              "cache_key": key, "source_sha256": digest,
              "model": model_name, "device": "cpu", "compute_type": "int8",
              "asr_library_version": asr_version,
              "duration": duration, "language": language,
              "language_probability": transcript.get("language_probability"),
              "segments": transcript["segments"], "candidates": candidates,
              "timing_provenance": TIMING_PROVENANCE,
              "transcript_path": str(path),
              "filtering_report": {"reason": reason,
                  "segment_count": len(transcript["segments"]),
                  "accepted_count": len(candidates), "rejected_count": len(rejections),
                  "rejection_counts": dict(Counter(item["reason"] for item in rejections)),
                  "rejections": rejections,
                  "thresholds": {"min_seconds": .4, "max_seconds": 8,
                      "min_words": 2, "max_words": 28,
                      "min_average_word_probability_exclusive": .75,
                      "max_no_speech_probability_exclusive": .35}},
              "limitations": ["ASR word timestamps are estimates.",
                              "No speaker identity is inferred.",
                              "Only English utterances are selected automatically."]}
    _write_json(path, result)
    if len(candidates) > maximum:
        raise ValueError(f"Source speech has {len(candidates)} candidates; limit is {maximum}. "
                         f"Supply a narrower source excerpt. Full transcript: {path}")
    return result
