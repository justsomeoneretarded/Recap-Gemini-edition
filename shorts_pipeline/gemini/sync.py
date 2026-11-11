"""Stage G: synchronize measured narration, dialogue, shots and captions.

Edit timing always comes from the synthesized audio files (measured duration
and per-word timings), never from word-count estimates. Each sentence's
footage is cut to that sentence's measured interval, retained source dialogue
keeps its original timing, and source dialogue under narration is ducked.
"""
from __future__ import annotations

from ..render import allocate_segmented, sentence_spans

NARRATION_DUCKED_GAIN = 0.015  # about -36 dB: source speech must not compete with narration.


def measured_sentence_seconds(beats: list[dict]) -> dict[str, float]:
    seconds = {}
    for beat in beats:
        if beat["kind"] != "narration" or not beat.get("words"):
            continue
        for sentence, (a, b) in zip(beat["sentences"], sentence_spans(beat)):
            seconds[sentence["sentence_id"]] = b - a
    return seconds


def duck_source_dialogue(beats: list[dict], segments: list[dict]) -> list[dict]:
    """Lower the source bed under narration wherever the source has speech."""
    notes = []
    speech = [(float(s["start"]), float(s["end"])) for s in segments
              if s.get("start") is not None and s.get("end") is not None]
    for beat in beats:
        if beat["kind"] != "narration":
            continue
        overlap = sum(max(0.0, min(b, shot["end"]) - max(a, shot["start"]))
                      for shot in beat.get("shots", []) for a, b in speech)
        if overlap > 0.2:
            beat["source_gain"] = NARRATION_DUCKED_GAIN
            notes.append({"beat_id": beat["id"], "source_speech_seconds_under_narration": round(overlap, 2),
                          "source_gain": NARRATION_DUCKED_GAIN})
    return notes


def dry_run_allocation(beats: list[dict], source_duration: float, fps: int) -> dict[str, list[dict]]:
    """Frame allocation exactly as the renderer will do it, or the error per beat."""
    result = {}
    for beat in beats:
        if beat["kind"] != "narration" or not beat.get("words"):
            continue
        try:
            result[beat["id"]] = allocate_segmented(beat, source_duration, fps)
        except ValueError as exc:
            result[beat["id"]] = [{"error": str(exc)}]
    return result


def timing_report(beats: list[dict]) -> dict:
    rows, total_words, total_seconds = [], 0, 0.0
    for beat in beats:
        row = {"beat_id": beat["id"], "kind": beat["kind"], "duration": round(beat.get("duration", 0), 3),
               "timing_provenance": beat.get("timing_provenance"), "words": beat.get("words", [])}
        if beat["kind"] == "narration" and beat.get("words"):
            row["sentences"] = [{"sentence_id": s["sentence_id"], "start": round(a, 3), "end": round(b, 3)}
                                for s, (a, b) in zip(beat["sentences"], sentence_spans(beat))]
            total_words += len(beat["words"])
            total_seconds += beat["duration"]
        rows.append(row)
    return {"measured_narration_words_per_second": round(total_words / total_seconds, 3) if total_seconds else None,
            "beats": rows}
