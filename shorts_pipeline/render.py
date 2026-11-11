"""Frame-bounded FFmpeg editing, audio mastering, and review artifacts.

The renderer consumes a reviewed storyboard or a model-generated, validated plan.
It never generates speech, invents source footage, loops a shot to cover missing
material, or uploads the result to a video platform.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def _run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Media command failed ({Path(args[0]).name}):\n{result.stderr[-5000:]}")
    return result


def _probe(path: Path) -> dict:
    data = json.loads(_run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]).stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    audio_start = float(audio.get("start_time", 0)) if audio else None
    audio_end = None
    if audio:
        if audio.get("duration") not in (None, "N/A"):
            audio_end = audio_start + float(audio["duration"])
        elif audio.get("tags", {}).get("DURATION"):
            hours, minutes, seconds = audio["tags"]["DURATION"].split(":")
            audio_end = int(hours)*3600 + int(minutes)*60 + float(seconds)
    return {"duration": float(data["format"]["duration"]), "width": video.get("width"),
            "height": video.get("height"), "has_audio": bool(audio),
            "audio_start": audio_start, "audio_end": audio_end,
            "streams": data.get("streams", [])}


def _require_audio_coverage(path: Path, start: float, end: float, tolerance: float = .065) -> None:
    """Verify retained dialogue has actual audio packets across its source span.

    Container duration or the presence of an audio stream cannot establish this:
    a four-second video may contain only half a second of source audio.
    """
    data = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-read_intervals",
                           f"{max(0., start-.1):.9f}%{end+.1:.9f}", "-show_packets", "-show_entries",
                           "packet=pts_time,duration_time", "-of", "json", str(path)]).stdout)
    ranges = []
    for packet in data.get("packets", []):
        if packet.get("pts_time") is None or packet.get("duration_time") is None:
            continue
        a = float(packet["pts_time"])
        b = a + float(packet["duration_time"])
        if b > start and a < end:
            ranges.append((max(start, a), min(end, b)))
    cursor = start
    for a, b in sorted(ranges):
        if a-cursor > tolerance:
            raise ValueError(f"Retained dialogue {start:.3f}–{end:.3f}s has missing source audio; choose a covered source range")
        cursor = max(cursor, b)
    if end-cursor > tolerance:
        raise ValueError(f"Retained dialogue {start:.3f}–{end:.3f}s has missing source audio; choose a covered source range")


def _ffmpeg(*args: str) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
            "-filter_threads", "1", "-filter_complex_threads", "1", *args]


def _number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def allocate_shots(shots: list[dict], duration: float, source_duration: float, fps: int) -> list[dict]:
    """Keep every selected visual anchor within an exact output-frame budget.

    Excess footage is trimmed around each observed midpoint by default. Reviewed
    shots may request start/end trims. A shortfall retains every source range
    and permits only a modest slowdown, never looping selected footage.
    """
    duration = _number(duration, "Beat duration")
    if duration <= 0 or not shots or fps <= 0:
        raise ValueError("Each beat needs a positive duration, frame rate and at least one source shot")
    clean = []
    for shot in shots:
        start, end = _number(shot.get("start"), "Shot start"), _number(shot.get("end"), "Shot end")
        focus = _number(shot.get("focus_x", .5), "Shot focus_x")
        if start < 0 or end <= start or end > source_duration + .001:
            raise ValueError(f"Source range {start:g}–{end:g}s is outside the {source_duration:g}s source")
        if not 0 <= focus <= 1:
            raise ValueError("focus_x must be between 0 and 1")
        trim_mode = shot.get("trim_mode", "center")
        if trim_mode not in ("start", "center", "end"):
            raise ValueError("trim_mode must be start, center or end")
        clean.append({**shot, "start": start, "end": end, "focus_x": focus, "trim_mode": trim_mode})
    target_frames = math.ceil(duration * fps - 1e-7)
    target = target_frames / fps
    spans = [s["end"] - s["start"] for s in clean]
    available = sum(spans)
    speed = min(1., available / target)
    if speed < .85 - 1e-6:
        raise ValueError(f"Only {available:.2f}s of selected footage for {duration:.2f}s narration; add source shots (minimum speed 0.85)")
    trimming = available > target + 1e-7
    # Full source spans in the slowdown case can end between output frames.
    # Their ceilings represent normal frame quantization, not repeated clips.
    capacities = [(math.floor(span * fps + 1e-7) if trimming else
                   math.ceil(span * fps / speed - 1e-7)) for span in spans]
    minimum = math.ceil(.25 * fps)
    lower = [min(minimum, capacity) if trimming else 1 for capacity in capacities]
    if any(capacity < 1 for capacity in capacities):
        raise ValueError("A selected source shot is shorter than one output frame")
    if sum(lower) > target_frames:
        raise ValueError("Too many selected shots for the narration frame budget; reduce shots or lengthen narration")
    if sum(capacities) < target_frames:
        raise ValueError("Selected footage cannot cover the narration within source frame bounds; add source shots")

    # Find the proportional scale with lower/upper bounds, then allocate
    # integer frames by largest remainder. No selected anchor disappears.
    low, high = 0., max(capacity / span for capacity, span in zip(capacities, spans))
    for _ in range(80):
        scale = (low + high) / 2
        quotas = [min(capacity, max(floor, scale * span))
                  for span, floor, capacity in zip(spans, lower, capacities)]
        if sum(quotas) < target_frames:
            low = scale
        else:
            high = scale
    quotas = [min(capacity, max(floor, high * span))
              for span, floor, capacity in zip(spans, lower, capacities)]
    frames = [math.floor(q + 1e-9) for q in quotas]
    remainder = target_frames - sum(frames)
    order = sorted(range(len(clean)), key=lambda i: (-(quotas[i] - frames[i]), i))
    for i in order:
        if remainder and frames[i] < capacities[i]:
            frames[i] += 1
            remainder -= 1
    if remainder:
        raise ValueError("Selected footage could not cover the narration")
    allocated = []
    for shot, span, count in zip(clean, spans, frames):
        source_span = count / fps if trimming else span
        spare = max(0., span - source_span)
        offset = {"start": 0., "center": spare / 2, "end": spare}[shot["trim_mode"]]
        start = shot["start"] + offset
        allocated.append({**shot, "source_start": shot["start"], "source_end": shot["end"],
                          "start": start, "end": min(shot["end"], start + source_span),
                          "frames": count, "duration": count / fps, "speed": speed})
    return allocated


def crop_filter(width: int, height: int, focus_x: float = .5) -> str:
    """Fill the requested canvas; focus_x locates the retained horizontal subject."""
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Canvas dimensions must be positive even integers")
    focus_x = _number(focus_x, "focus_x")
    if not 0 <= focus_x <= 1:
        raise ValueError("focus_x must be between 0 and 1")
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:x='(in_w-out_w)*{focus_x:.6f}':y='(in_h-out_h)/2',setsar=1")


def _concat(files: list[Path], target: Path) -> None:
    # Only internally generated names are written to the concat manifest.
    manifest = target.with_suffix(".concat.txt")
    manifest.write_text("".join(f"file '{p.name}'\n" for p in files), encoding="utf-8")
    _run(_ffmpeg("-f", "concat", "-safe", "1", "-i", str(manifest), "-map", "0:v:0", "-map", "0:a:0",
                 "-c", "copy", str(target)))


def _render_shot(source: Path, shot: dict, target: Path, width: int, height: int, fps: int, has_audio: bool) -> None:
    speed = shot["speed"]
    inputs = ["-ss", f"{shot['start']:.9f}", "-t", f"{shot['end']-shot['start']:.9f}", "-i", str(source)]
    # The source bed always has an explicit, finite clock. Missing effects/music
    # in a narration shot become zero-valued samples; they cannot truncate voice.
    # Retained dialogue is separately checked for actual source-packet coverage.
    inputs += ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={shot['duration']:.9f}"]
    vf = f"{crop_filter(width, height, shot['focus_x'])},setpts=(PTS-STARTPTS)/{speed:.9f},fps={fps},format=yuv420p"
    silence = f"[1:a]atrim=duration={shot['duration']:.9f},asetpts=PTS-STARTPTS[bed]"
    if has_audio:
        af = (silence + f";[0:a]aresample=48000:async=1:first_pts=0,atempo={speed:.9f},"
              "aformat=channel_layouts=stereo[source];"
              "[bed][source]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    else:
        af = silence.replace("[bed]", "[a]")
    _run(_ffmpeg(*inputs, "-map", "0:v:0", "-map", "[a]",
                 "-vf", vf, "-filter_complex", af, "-frames:v", str(shot["frames"]), "-t", f"{shot['duration']:.9f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "1", "-c:a", "pcm_s16le", str(target)))


def _attach_narration(footage: Path, audio: Path, target: Path, source_gain: float, duration: float) -> None:
    # Narration anchors the mix. Only the sub-frame tail is padded; finite source
    # beds or sparse source sound never determine how much speech survives.
    filters = (f"[0:a]volume={source_gain:.6f},apad=whole_dur={duration:.9f},atrim=duration={duration:.9f}[source];"
               "[1:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,"
               f"apad=whole_dur={duration:.9f},atrim=duration={duration:.9f}[voice];"
               "[voice][source]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]")
    _run(_ffmpeg("-i", str(footage), "-i", str(audio), "-filter_complex", filters,
                 "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "pcm_s16le", str(target)))


def _master_audio(assembled: Path, target: Path, music: Path | None) -> dict:
    inputs = ["-i", str(assembled)]
    if music:
        # A supplied music track may loop; footage and narration never do.
        edit_duration = _number(_probe(assembled)["duration"], "Edit duration")
        music_duration = _number(_probe(music)["duration"], "Music duration")
        if music_duration <= 0:
            raise ValueError("Music duration must be positive")
        repeats = max(0, math.ceil(edit_duration / music_duration) - 1)
        inputs += ["-stream_loop", str(repeats), "-t", f"{edit_duration:.9f}", "-i", str(music)]
        # Keep every branch alive through the edit. FFmpeg 6 can otherwise
        # propagate the loop/sidechain EOF through asplit and cut the voice tail.
        filters = (f"[0:a]asplit=2[main][side];[1:a]volume=0.10,aresample=48000,"
                   f"apad=whole_dur={edit_duration:.9f},atrim=duration={edit_duration:.9f}[music];"
                   "[music][side]sidechaincompress=threshold=0.02:ratio=8:attack=10:release=250,"
                   f"apad=whole_dur={edit_duration:.9f},atrim=duration={edit_duration:.9f}[duck];"
                   "[main][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")
        _run(_ffmpeg(*inputs, "-filter_complex", filters, "-map", "0:v:0", "-map", "[mix]",
                     "-c:v", "copy", "-c:a", "pcm_s16le", str(target)))
    else:
        _run(_ffmpeg(*inputs, "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(target)))
    measurement = _run(["ffmpeg", "-hide_banner", "-nostats", "-threads", "1", "-i", str(target),
                        "-vn", "-af", "loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json", "-f", "null", "-"])
    blocks = re.findall(r'\{\s*"input_i".*?\}', measurement.stderr, re.S)
    if not blocks:
        raise RuntimeError("FFmpeg did not return loudness measurements")
    data = json.loads(blocks[-1])
    # Digital silence produces -inf; it must not masquerade as mastered speech.
    if any(not math.isfinite(float(data[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
        raise ValueError("Output audio is silent or cannot be loudness-normalized")
    return data


def _ass_escape(path: Path) -> str:
    # Escape the FFmpeg filter parser only, never invoke a shell.
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")


def _covers(video: Path, outdir: Path, duration: float) -> dict:
    from PIL import Image, ImageDraw
    paths = []
    for index, fraction in enumerate((.12, .5, .82), 1):
        path = outdir / f"cover-{index:02d}.jpg"
        _run(_ffmpeg("-ss", f"{max(0, min(duration-.05, duration*fraction)):.6f}", "-i", str(video),
                     "-frames:v", "1", "-q:v", "2", "-threads", "1", str(path)))
        paths.append(path)
    grid = Image.new("RGB", (4 * 270, 2 * 300), (18, 18, 18))
    draw = ImageDraw.Draw(grid)
    with tempfile.TemporaryDirectory(prefix="contact-", dir=outdir) as temp:
        for index in range(8):
            at = min(duration-.05, duration*(index+.5)/8)
            frame = Path(temp) / f"frame-{index}.jpg"
            _run(_ffmpeg("-ss", f"{max(0, at):.6f}", "-i", str(video), "-frames:v", "1",
                         "-vf", "scale=270:270:force_original_aspect_ratio=decrease,pad=270:270:(ow-iw)/2:(oh-ih)/2",
                         "-threads", "1", str(frame)))
            with Image.open(frame) as image:
                grid.paste(image.convert("RGB"), ((index % 4)*270, (index//4)*300))
            draw.text(((index % 4)*270+8, (index//4)*300+276), f"{at:.2f}s", fill="white")
    contact = outdir / "contact-sheet.jpg"
    grid.save(contact, quality=90)
    return {"covers": paths, "contact_sheet": contact}


def render_plan(plan: dict, source: Path, outdir: Path, width: int = 1080, height: int = 1080,
                fps: int = 60, music: Path | None = None) -> dict:
    """Render and verify a complete plan; return artifact paths and QC metadata."""
    source, outdir = Path(source).resolve(), Path(outdir).resolve()
    if fps not in (24, 25, 30, 50, 60):
        raise ValueError("Unsupported output frame rate")
    crop_filter(width, height)  # Validate dimensions before processing media.
    if not source.is_file():
        raise FileNotFoundError(source)
    if music is not None:
        music = Path(music).resolve()
        if not music.is_file() or not _probe(music)["has_audio"]:
            raise ValueError("Music must be an existing audio file")
    source_info = _probe(source)
    if not source_info["width"]:
        raise ValueError("Source has no video stream")
    beats = copy.deepcopy(plan.get("beats", []))
    if not beats:
        raise ValueError("The storyboard has no beats")
    target_time = 0.
    frame_time = 0.
    for index, beat in enumerate(beats):
        kind = beat.get("kind", "narration")
        if kind not in ("narration", "dialogue"):
            raise ValueError(f"Unknown beat kind: {kind}")
        shots = beat.get("shots", [])
        if kind == "dialogue":
            if len(shots) != 1 or not source_info["has_audio"]:
                raise ValueError("Dialogue requires one source shot and a source audio stream")
            duration = _number(shots[0].get("end"), "Dialogue end") - _number(shots[0].get("start"), "Dialogue start")
            _require_audio_coverage(source, float(shots[0]["start"]), float(shots[0]["end"]))
            if beat.get("duration") is not None and abs(_number(beat["duration"], "Dialogue duration") - duration) > .03:
                raise ValueError("Dialogue duration must equal its source range")
        else:
            audio = Path(beat.get("audio_path", "")).resolve()
            if not audio.is_file():
                raise ValueError(f"Narration beat {beat.get('id', index)} needs a synthesized or supplied audio_path")
            audio_info = _probe(audio)
            if not audio_info["has_audio"]:
                raise ValueError("Narration file has no audio stream")
            duration = audio_info["duration"]
            if beat.get("duration") is not None and abs(_number(beat["duration"], "Narration duration") - duration) > .08:
                raise ValueError("Narration duration does not match the actual audio file")
            beat["audio_path"] = str(audio)
        if beat.get("timeline_start") is not None and abs(_number(beat["timeline_start"], "timeline_start")-target_time) > .05:
            raise ValueError("Storyboard timeline contains a gap or overlap")
        beat.update(duration=duration, timeline_start=target_time, timeline_end=target_time+duration)
        beat["render_shots"] = allocate_shots(shots, duration, source_info["duration"], fps)
        # Dialogue must retain its original speed. A sub-frame ceiling is
        # handled by normal frame quantization, not an intentional retime.
        if kind == "dialogue":
            for shot in beat["render_shots"]:
                shot["speed"] = 1.
        target_time += duration
        beat["render_timeline_start"] = frame_time
        frame_time += sum(s["duration"] for s in beat["render_shots"])
        beat["render_timeline_end"] = frame_time
    if target_time > 180 + .001 or frame_time > 180 + .001:
        raise ValueError(f"Storyboard is {target_time:.2f}s; shorten the script to at most 180s")
    if frame_time - target_time > .25:
        raise ValueError("Too many sub-frame beat boundaries; combine very short narration beats before rendering")
    outdir.mkdir(parents=True, exist_ok=True)
    final = outdir / "final.mp4"
    with tempfile.TemporaryDirectory(prefix="render-", dir=outdir) as temp:
        work = Path(temp)
        rendered = []
        for beat_index, beat in enumerate(beats):
            shots = []
            for shot_index, shot in enumerate(beat["render_shots"]):
                clip = work / f"b{beat_index:03d}-s{shot_index:03d}.mkv"
                _render_shot(source, shot, clip, width, height, fps, source_info["has_audio"])
                shots.append(clip)
            footage = work / f"b{beat_index:03d}-footage.mkv"
            _concat(shots, footage)
            if beat.get("kind", "narration") == "narration":
                result = work / f"b{beat_index:03d}-voiced.mkv"
                gain = _number(beat.get("source_gain", .04), "Source audio gain")
                if not 0 <= gain <= .15:
                    raise ValueError("Narrated source_gain must be between 0 and 0.15")
                _attach_narration(footage, Path(beat["audio_path"]), result, gain,
                                  sum(s["duration"] for s in beat["render_shots"]))
            else:
                result = footage
            rendered.append(result)
        assembled = work / "assembled.mkv"
        _concat(rendered, assembled)
        mixed = work / "mixed.mkv"
        loudness = _master_audio(assembled, mixed, music)
        af = ("loudnorm=I=-14:TP=-1.5:LRA=9:"
              f"measured_I={loudness['input_i']}:measured_TP={loudness['input_tp']}:"
              f"measured_LRA={loudness['input_lra']}:measured_thresh={loudness['input_thresh']}:"
              f"offset={loudness['target_offset']}:linear=true,aresample=48000")
        args = ["-i", str(mixed), "-map", "0:v:0", "-map", "0:a:0"]
        captions = outdir / "captions.ass"
        if captions.is_file():
            args += ["-vf", f"ass=filename='{_ass_escape(captions)}'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "1"]
        else:
            args += ["-c:v", "copy"]
        pending = work / "verified-final.mp4"
        _run(_ffmpeg(*args, "-af", af, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(pending)))
        actual = _probe(pending)
        if (actual["width"], actual["height"]) != (width, height) or not actual["has_audio"]:
            raise RuntimeError("Rendered media failed resolution/audio checks")
        audio_end = actual["audio_end"]
        last_voice_endpoint = max((b["render_timeline_start"]+b["duration"] for b in beats
                                   if b.get("kind", "narration") == "narration"), default=0.)
        if audio_end is None or audio_end < last_voice_endpoint-.02 or audio_end < frame_time-.065:
            raise RuntimeError(f"Rendered audio ends before the planned voice/timeline endpoint: "
                               f"audio={audio_end!r}s, voice={last_voice_endpoint:.6f}s, edit={frame_time:.6f}s")
        if abs(actual["duration"] - frame_time) > .18:
            raise RuntimeError(f"Rendered duration {actual['duration']:.3f}s differs from planned {frame_time:.3f}s")
        if actual["duration"] > 180.001:
            raise RuntimeError("Encoded output exceeds 180 seconds; shorten the final beat before exporting")
        os.replace(pending, final)
    qc = {"status": "passed", "duration_seconds": actual["duration"], "narration_timeline_seconds": target_time,
          "frame_timeline_seconds": frame_time, "frame_rounding_seconds": frame_time-target_time,
          "width": width, "height": height, "fps": fps, "has_audio": True,
          "audio_end_seconds": actual["audio_end"], "last_voice_endpoint_seconds": last_voice_endpoint,
          "source_audio_bed_policy": "Missing source-bed audio is zero-filled under narration; retained dialogue requires packet coverage.",
          "captions_burned": (outdir/"captions.ass").is_file(),
          "loudness_target": {"integrated_lufs": -14, "true_peak_dbfs": -1.5, "range_lu": 9},
          "input_mix_measurement": loudness,
          "checks": ["source ranges", "speech duration", "footage coverage", "timeline continuity", "output geometry", "output duration", "audio endpoint", "retained dialogue audio coverage"],
          "limitations": ["Loudness target applied with measured two-pass normalization; final AAC output is not independently loudness-certified.",
                          "Cover candidates are timeline samples; human review is needed for cover selection."]}
    timeline = outdir/"timeline.json"
    timeline.write_text(json.dumps({**plan, "beats": beats}, indent=2), encoding="utf-8")
    provenance = outdir/"source-attribution.json"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024*1024):
            digest.update(chunk)
    provenance.write_text(json.dumps({"source_name": source.name, "source_sha256": digest.hexdigest(),
                                    "source_origin": plan.get("source_origin"), "original_source_window": plan.get("source_window"),
                                    "title": plan.get("title", ""), "source_ranges": [
                                        {"beat": b.get("id", i), "kind": b.get("kind", "narration"), "shots": b["render_shots"]}
                                        for i,b in enumerate(beats)]}, indent=2), encoding="utf-8")
    qc_path = outdir/"quality-report.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    artifacts = _covers(final, outdir, actual["duration"])
    return {"video": final, "final": final, "timeline": timeline, "provenance": provenance,
            "quality_report": qc_path, "qc": qc, **artifacts}
