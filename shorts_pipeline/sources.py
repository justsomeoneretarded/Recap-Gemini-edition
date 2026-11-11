"""Acquire footage and index shot boundaries without inventing source material.

Only explicit local files, direct HTTPS media and supported YouTube URLs are
accepted. A movie title is not enough to obtain the underlying movie footage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

INDEX_VERSION = 1
MAX_SHOT_SECONDS = 6.0


def _run(args: list[str], *, timeout: float = 900) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable is missing: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{args[0]} exceeded its {timeout:g}-second timeout.") from exc
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-2500:].strip()}")
    return result


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_info(path: Path) -> dict:
    response = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                     "-of", "json", str(path)], timeout=60)
    try:
        data = json.loads(response.stdout)
        video = next(s for s in data["streams"] if s.get("codec_type") == "video")
        duration = float(data.get("format", {}).get("duration") or video.get("duration"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("non-positive duration")
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise RuntimeError(f"Source does not contain a readable finite video: {path}") from exc
    return {"duration": duration, "width": int(video["width"]), "height": int(video["height"]),
            "has_audio": any(s.get("codec_type") == "audio" for s in data["streams"])}


def _https_url(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    return True


def _youtube(value: str) -> bool:
    host = (urlsplit(value).hostname or "").lower()
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def resolve_source(value: str, workdir: Path) -> Path:
    """Resolve an explicit footage input; return a validated local video path.

    Downloads are written atomically. The source manifest records the original
    URL, actual byte count and SHA-256; failed downloads never become cache hits.
    """
    workdir = Path(workdir)
    directory = workdir / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    if not value or not isinstance(value, str):
        raise ValueError("Provide a source video file or HTTPS video URL; a film title alone is insufficient.")
    value = value.strip()
    parsed = urlsplit(value)
    if not parsed.scheme:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source video not found: {path}. Supply footage or a direct HTTPS video URL.")
        info = _video_info(path)
        _write_json(directory / "source.json", {"origin": str(path), "path": str(path),
                    "sha256": file_digest(path), "bytes": path.stat().st_size, **info})
        return path
    if not _https_url(value):
        raise ValueError("Remote footage must use an HTTPS URL without embedded credentials.")

    key = hashlib.sha256(value.encode()).hexdigest()[:20]
    destination = directory / f"source-{key}.mp4"
    manifest_path = directory / f"source-{key}.json"
    if destination.is_file() and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if previous.get("origin") == value and previous.get("sha256") == file_digest(destination):
                _video_info(destination)
                _write_json(directory / "source.json", previous)
                return destination
        except (OSError, ValueError, RuntimeError):
            pass

    maximum = int(os.getenv("SHORTS_MAX_DOWNLOAD_BYTES", str(2 * 1024 ** 3)))
    if maximum < 1:
        raise ValueError("SHORTS_MAX_DOWNLOAD_BYTES must be positive.")
    with tempfile.TemporaryDirectory(prefix="download-", dir=directory) as temp:
        temp_path = Path(temp)
        downloaded = temp_path / "video.mp4"
        if _youtube(value):
            try:
                import yt_dlp
            except ImportError as exc:
                raise RuntimeError("YouTube downloads need yt-dlp. Install recap-shorts[source] or use a local/direct video file.") from exc
            try:
                with yt_dlp.YoutubeDL({
                    "outtmpl": str(temp_path / "video.%(ext)s"),
                    "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                    "merge_output_format": "mp4", "noplaylist": True,
                    "socket_timeout": 30, "retries": 2, "fragment_retries": 2,
                    "max_filesize": maximum, "cachedir": False,
                    "quiet": True, "no_warnings": False,
                }) as downloader:
                    metadata = downloader.extract_info(value, download=True)
                    candidate = Path(downloader.prepare_filename(metadata))
                candidates = [downloaded, candidate, *temp_path.glob("video.*")]
                downloaded = next(p for p in candidates if p.is_file() and p.suffix in {".mp4", ".webm", ".mkv", ".mov"})
            except Exception as exc:
                raise RuntimeError("YouTube footage could not be downloaded. Use a local source file or a direct HTTPS media URL "
                                   f"(such as a GitHub Release attachment). Downloader: {exc}") from exc
        else:
            import requests
            try:
                with requests.get(value, stream=True, timeout=(20, 60),
                                  headers={"User-Agent": "RecapShorts/0.1"}) as response:
                    response.raise_for_status()
                    length = int(response.headers.get("Content-Length") or 0)
                    if length > maximum:
                        raise ValueError(f"Source exceeds the configured {maximum}-byte download limit.")
                    received = 0
                    with downloaded.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > maximum:
                                raise ValueError(f"Source exceeds the configured {maximum}-byte download limit.")
                            output.write(chunk)
            except requests.RequestException as exc:
                raise RuntimeError(f"Could not download source video: {exc}. Supply a reachable direct media URL or local video.") from exc
        if downloaded.stat().st_size > maximum:
            raise ValueError(f"Downloaded footage exceeds {maximum} bytes.")
        info = _video_info(downloaded)
        digest = file_digest(downloaded)
        # Keep the original container bytes; FFmpeg inspects the stream rather
        # than relying on the .mp4 suffix for downloaded webm/mkv fallbacks.
        shutil.move(str(downloaded), destination)
    _write_json(manifest_path, {"origin": value, "path": str(destination.resolve()),
                "sha256": digest, "bytes": destination.stat().st_size, **info})
    _write_json(directory / "source.json", json.loads(manifest_path.read_text()))
    return destination.resolve()


def _adaptive_boundaries(path: Path, fps: float) -> list[float]:
    from scenedetect import AdaptiveDetector, detect
    minimum = max(1, round(fps * .35))
    found = detect(str(path), AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=minimum),
                   show_progress=False, start_in_scene=True)
    return [float(start.get_seconds()) for start, _ in found]


def _ffmpeg_boundaries(path: Path, duration: float) -> list[float]:
    """Threshold candidates, explicitly labeled as an approximate fallback."""
    response = _run([
        "ffmpeg", "-hide_banner", "-nostdin", "-threads", "1", "-i", str(path),
        "-an", "-vf", "scale=320:-2,select='gt(scene,0.25)',showinfo",
        "-filter_threads", "1", "-f", "null", "-",
    ], timeout=max(120, min(7200, duration * 5)))
    raw = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", response.stderr)]
    # Consecutive-frame triggers can arise during flashes or rapid motion.
    accepted = []
    for t in raw:
        if t >= .35 and (not accepted or t - accepted[-1] >= .35):
            accepted.append(t)
    return accepted


def _segments(boundaries: list[float], duration: float, max_seconds: float = MAX_SHOT_SECONDS) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Video duration and maximum segment length must be positive and finite.")
    clean = sorted({0.0, duration, *(float(t) for t in boundaries if math.isfinite(float(t)) and 0 < float(t) < duration)})
    result = []
    for start, end in zip(clean, clean[1:]):
        count = max(1, math.ceil((end - start) / max_seconds))
        for index in range(count):
            a = start + (end - start) * index / count
            b = start + (end - start) * (index + 1) / count
            if b - a > .001:
                result.append((a, b))
    return result


def index_source(path: Path, workdir: Path) -> list[dict]:
    """Index all source footage, subdividing long shots into <=6-second spans.

    Every returned span has one midpoint JPEG. These images are representative
    samples, not proof that every motion/action within a shot was observed.
    """
    path, workdir = Path(path).resolve(), Path(workdir)
    info = _video_info(path)
    checksum = file_digest(path)
    directory = workdir / "index" / checksum[:20]
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "index.json"
    if cache.exists():
        try:
            previous = json.loads(cache.read_text())
            if (previous.get("version") == INDEX_VERSION and previous.get("source_sha256") == checksum
                    and all(Path(s["frame_path"]).is_file() for s in previous["scenes"])):
                _write_json(workdir / "source-index.json", previous)
                return previous["scenes"]
        except (ValueError, KeyError, TypeError):
            pass
    details = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=avg_frame_rate", "-of", "json", str(path)], timeout=60)
    ratio = json.loads(details.stdout)["streams"][0].get("avg_frame_rate", "30/1")
    numerator, denominator = ratio.split("/")
    fps = float(numerator) / max(1, float(denominator))
    fallback_reason = None
    try:
        boundaries = _adaptive_boundaries(path, fps)
        method = "PySceneDetect AdaptiveDetector (adaptive_threshold=3.0, minimum=.35s)"
    except (ImportError, ModuleNotFoundError) as exc:
        fallback_reason = type(exc).__name__
        boundaries = _ffmpeg_boundaries(path, info["duration"])
        method = "FFmpeg scene-score >0.25 at 320px width, .35s debounce; approximate candidates"
    spans = _segments(boundaries, info["duration"])
    maximum = int(os.getenv("SHORTS_MAX_INDEX_SCENES", "2000"))
    if len(spans) > maximum:
        raise ValueError(f"Source contains {len(spans)} indexed spans, exceeding SHORTS_MAX_INDEX_SCENES={maximum}. "
                         "Provide a relevant source excerpt, or deliberately increase the indexing limit.")
    scenes = []
    for number, (start, end) in enumerate(spans, 1):
        frame = directory / f"scene-{number:05d}.jpg"
        midpoint = (start + end) / 2
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
              "-ss", f"{midpoint:.6f}", "-i", str(path), "-frames:v", "1",
              "-vf", "scale=640:-2", "-threads", "1", "-q:v", "3", "-y", str(frame)], timeout=90)
        if not frame.is_file() or frame.stat().st_size == 0:
            raise RuntimeError(f"No representative frame could be extracted at {midpoint:.3f}s.")
        scenes.append({"id": f"S{number:04d}", "start": round(start, 6), "end": round(end, 6),
                       "frame_path": str(frame.resolve()), "sample_time": round(midpoint, 6)})
    _write_json(cache, {"version": INDEX_VERSION, "source_path": str(path), "source_sha256": checksum,
                       "duration": info["duration"], "method": method, "fallback_reason": fallback_reason,
                       "coverage": "All source time ranges indexed; one midpoint image per <=6-second span.",
                       "scenes": scenes})
    _write_json(workdir / "source-index.json", json.loads(cache.read_text()))
    return scenes
