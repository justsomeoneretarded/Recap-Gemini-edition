from __future__ import annotations
import json
import hashlib
import math
import subprocess
from pathlib import Path

EXCERPT_VERSION = 1


def run(args: list[str], *, timeout: float = 1800) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {args[0]}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-3000:]}")
    return result


def probe(path: Path) -> dict:
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]).stdout)
    video = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    seconds = float(data['format']['duration'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Media has no positive finite duration.')
    result = {'duration': seconds, 'has_audio': any(s['codec_type'] == 'audio' for s in data['streams'])}
    if video:
        a, b = video.get('avg_frame_rate', '30/1').split('/')
        result.update(width=video['width'], height=video['height'], fps=float(a)/max(1,float(b)))
    return result


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def excerpt(source: Path, workdir: Path, window: list[float]) -> Path:
    info=probe(source)
    start,end=window
    if end>info['duration']+.02:
        raise ValueError(f'Source window ends at {end}s, beyond the {info["duration"]:.2f}s source.')
    with source.open('rb') as stream:
        fingerprint=hashlib.file_digest(stream,'sha256').hexdigest()
    identity=hashlib.sha256(f'{EXCERPT_VERSION}:{fingerprint}:{start}:{end}'.encode()).hexdigest()[:20]
    # Persist the canonical encoded excerpt with the source cache. Re-encoding
    # on different runner CPUs can change its bytes and invalidate scene/vision
    # caches despite using the same original footage and time window.
    directory=Path(workdir)/'sources'/'excerpts'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'excerpt-{identity}.mp4'
    if not target.exists():
        pending=target.with_name(target.stem+'.pending.mp4')
        run(['ffmpeg','-y','-v','error','-threads','1','-filter_threads','1',
             '-ss',str(start),'-i',str(source),'-t',str(end-start),'-map','0:v:0','-map','0:a:0?',
             '-c:v','libx264','-preset','veryfast','-crf','18','-threads','1',
             '-c:a','aac','-b:a','192k',str(pending)])
        pending.replace(target)
    if abs(probe(target)['duration']-(end-start))>.15:
        raise RuntimeError('Prepared source excerpt failed its duration check.')
    return target
