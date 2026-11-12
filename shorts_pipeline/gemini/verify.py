"""Stage J: measurements of the actual rendered MP4 (not the plan).

Checks duration, geometry, frame rate and frame count, audio/video end
alignment, loudness/true peak/clipping, silence inside narration and blank
tails, black and frozen sections, caption cue timing, repeated source ranges
and source-range validity. An optional Gemini pass watches the final file when
the preset asks for it or when automatic checks find issues. The result never
declares a video production-ready: human playback review is always required.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .client import text_part, video_part
from .schemas import QualityReview

SILENCE_DB = -45
SILENCE_SECONDS = 0.6


def _ffmpeg_filter(path: Path, args: list[str]) -> str:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), *args, "-f", "null", "-"],
                            capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"ffmpeg measurement failed: {result.stderr[-1500:]}")
    return result.stderr


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_streams", "-show_format", "-of", "json",
                          str(path)], capture_output=True, text=True, timeout=300, check=True).stdout
    return json.loads(out)


def _ranges(text: str, start_key: str, end_key: str) -> list[tuple[float, float]]:
    starts = [float(x) for x in re.findall(start_key + r"[:=]\s*(-?[0-9.]+)", text)]
    ends = [float(x) for x in re.findall(end_key + r"[:=]\s*(-?[0-9.]+)", text)]
    return list(zip(starts, ends))


def _srt(path: Path) -> list[tuple[float, float]]:
    def seconds(value):
        h, m, rest = value.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    if not path.is_file():
        return []
    return [(seconds(a), seconds(b)) for a, b in re.findall(r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)", path.read_text())]


def verify_render(final: Path, timeline: dict, outdir: Path, *, width: int, height: int, fps: int,
                  target: float, source_duration: float, client=None, render_review=False, log=print) -> dict:
    final, outdir = Path(final), Path(outdir)
    checks, issues = {}, []

    def fail(name, severity, detail):
        issues.append({"check": name, "severity": severity, "description": detail})

    data = _probe(final)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    duration = float(data["format"]["duration"])
    num, den = video.get("avg_frame_rate", "0/1").split("/")
    actual_fps = float(num) / max(1.0, float(den))
    frames = int(video.get("nb_read_packets") or 0)
    checks["container"] = {"duration": duration, "width": video["width"], "height": video["height"],
                           "fps": round(actual_fps, 3), "frames": frames, "has_audio": audio is not None}
    if (video["width"], video["height"]) != (width, height):
        fail("dimensions", "critical", f"{video['width']}x{video['height']} instead of {width}x{height}")
    if abs(actual_fps - fps) > 0.05:
        fail("fps", "critical", f"{actual_fps:.3f} fps instead of {fps}")
    planned = sum(float(b["duration"]) for b in timeline["beats"])
    if abs(duration - planned) > 0.2:
        fail("duration", "critical", f"rendered {duration:.2f}s but the measured timeline is {planned:.2f}s")
    if duration > min(180.0, target) + 0.25:
        fail("duration", "critical", f"rendered {duration:.2f}s exceeds the {target:.0f}s maximum")
    video_seconds = float(video.get("duration") or duration)
    if frames and abs(frames - video_seconds * fps) > 2:
        fail("frame_count", "major", f"{frames} frames for {video_seconds:.3f}s of video at {fps} fps")
    if audio is None:
        fail("audio", "critical", "no audio stream")
    else:
        audio_end = float(audio.get("start_time", 0)) + float(audio.get("duration", duration))
        checks["audio_video_end_difference"] = round(audio_end - float(video.get("duration", duration)), 4)
        if abs(checks["audio_video_end_difference"]) > 0.1:
            fail("av_alignment", "major", f"audio and video end {checks['audio_video_end_difference']:+.3f}s apart")
        loud = _ffmpeg_filter(final, ["-vn", "-af", "ebur128=peak=true"])
        integrated = re.findall(r"I:\s+(-?[0-9.]+) LUFS", loud)
        peak = re.findall(r"Peak:\s+(-?[0-9.inf]+) dBFS", loud)
        checks["loudness"] = {"integrated_lufs": float(integrated[-1]) if integrated else None,
                              "true_peak_dbtp": float(peak[-1]) if peak and peak[-1] not in ("-inf", "inf") else None}
        lufs, tp = checks["loudness"]["integrated_lufs"], checks["loudness"]["true_peak_dbtp"]
        if lufs is None or abs(lufs + 14) > 1.5:
            fail("loudness", "major", f"integrated loudness {lufs} LUFS; target -14")
        if tp is not None and tp > -1.0:
            fail("clipping", "major" if tp <= 0 else "critical", f"true peak {tp:.2f} dBTP (ceiling -1.5)")
        silence = _ranges(_ffmpeg_filter(final, ["-vn", "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_SECONDS}"]),
                          "silence_start", "silence_end")
        checks["silences"] = silence
        speech = []
        for beat in timeline["beats"]:
            start = float(beat.get("render_timeline_start", beat.get("timeline_start", 0)))
            for word in beat.get("words", []):
                speech.append((start + float(word["start"]), start + float(word["end"])))
        for a, b in silence:
            inside = sum(max(0, min(b, wb) - max(a, wa)) for wa, wb in speech)
            if inside > 0.3:
                fail("speech_continuity", "major", f"silence {a:.2f}–{b:.2f}s overlaps {inside:.2f}s of timed words")
        if speech:
            last = max(b for _, b in speech)
            checks["speech_tail_seconds"] = round(duration - last, 3)
            if last > duration + 0.05:
                fail("speech_tail", "critical", f"last word ends at {last:.2f}s after the video ends")
            elif duration - last > 2.5:
                fail("speech_tail", "minor", f"{duration - last:.2f}s after the final word")
    black = _ranges(_ffmpeg_filter(final, ["-an", "-vf", "blackdetect=d=0.3:pix_th=0.10"]), "black_start", "black_end")
    frozen = _ranges(_ffmpeg_filter(final, ["-an", "-vf", "freezedetect=n=0.003:d=2.5"]),
                     "lavfi.freezedetect.freeze_start", "lavfi.freezedetect.freeze_end")
    checks["black_sections"], checks["frozen_sections"] = black, frozen
    for a, b in black:
        fail("black_frames", "major", f"black video {a:.2f}–{b:.2f}s")
    for a, b in frozen:
        fail("frozen_video", "minor", f"no picture change {a:.2f}–{b:.2f}s; confirm it is a deliberately still shot")
    cues = _srt(outdir / "captions.srt")
    checks["caption_cues"] = len(cues)
    if any(b <= a for a, b in cues) or any(n[0] < p[0] for p, n in zip(cues, cues[1:])) or (cues and cues[-1][1] > duration + 0.05):
        fail("captions", "major", "caption cues are unordered, empty or extend past the video")
    used = []
    for beat in timeline["beats"]:
        for shot in beat.get("render_shots", []):
            a, b = float(shot["start"]), float(shot["end"])
            if not 0 <= a < b <= source_duration + 1e-3:
                fail("source_range", "critical", f"beat {beat['id']} uses {a:.2f}–{b:.2f}s outside the source")
            for ua, ub, owner in used:
                if min(b, ub) - max(a, ua) > 0.25 and owner != shot.get("sentence_id", beat["id"]):
                    fail("repeated_footage", "major", f"{a:.2f}–{b:.2f}s repeats footage used by {owner}")
            used.append((a, b, shot.get("sentence_id", beat["id"])))
    reviewed = None
    if client is not None and (render_review is True or (render_review == "on_issues" and issues)):
        try:
            file = client.upload(final)
            log("Gemini: reviewing the rendered video")
            result = client.generate(
                "review:render", QualityReview,
                [video_part(file, fps=1.0), text_part(
                    "This is the finished recap. Watch it with sound. Report concrete problems only: narration that "
                    "does not match the picture at that moment, unreadable or mistimed captions, abrupt or confusing "
                    "cuts, audio problems, black/frozen picture, a weak opening or an ending that does not land. "
                    "Give timeline seconds in source_start/source_end.")],
                system="You are a strict final-QC reviewer of a rendered video. You approve nothing by default.",
                temperature=0.0, max_output_tokens=16384, resolution="low", thinking="medium")
            reviewed = result.model_dump()
            for item in result.issues:
                issues.append({"check": "gemini_render_review:" + item.check, "severity": item.severity,
                               "description": item.description, "timeline_start": item.source_start,
                               "timeline_end": item.source_end})
        except Exception as exc:  # noqa: BLE001 - the measurement report must still be written
            reviewed = {"error": str(exc)[:500]}
    verdict = ("failed" if any(i["severity"] == "critical" for i in issues)
               else "passed_with_warnings" if issues else "passed")
    report = {"automated_verdict": verdict, "production_ready": False,
              "human_review_required": True,
              "human_review_checklist": ["Watch the whole video with sound at phone size.",
                                         "Confirm each narrated action is visible while it is spoken.",
                                         "Confirm names, quotes and story facts against the film.",
                                         "Check caption words, line breaks and obstruction of faces/objects.",
                                         "Audition voice, retained dialogue and mix balance."],
              "checks": checks, "issues": issues, "gemini_render_review": reviewed}
    (outdir / "final-verification.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
