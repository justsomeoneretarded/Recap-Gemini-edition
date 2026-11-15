"""Find recap footage on YouTube when no source video is supplied (`source: discover`).

1. Story plan: Gemini lists the film's famous or action scenes in story order, plus the
   connected scenes the story needs around each one, and broad search queries for them.
2. Search: each query is searched on YouTube (metadata only, nothing downloaded).
   Trailers, recaps, reactions, edits and similar uploads are dropped by title.
3. Rank: Gemini picks the candidates most likely to contain the story items.
4. Screen: Gemini watches each picked video from its link (no download) and marks the
   segments that show a story item, rejecting the wrong film or unclean footage
   (narration, burned-in captions, watermarks, mirrored or sped-up copies).
5. Coverage: must-have items with no footage get one more search round.
6. Download only the approved segments (plus a few seconds of handles), normalize them,
   and join them in story order into one footage pool for the normal pipeline.

Everything is cached under workdir/discovery/<key>, and discovery-report.json lists
every story item, query, verdict and chosen segment.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from .client import CostLedger, GeminiClient, _write_json, text_part
from .settings import settings_from_request

# Uploads that are not usable source footage for a recap.
REJECT_TITLE = re.compile(
    r"\b(trailer|teaser|recap|recapped|explained|explanation|reaction|reacts?|review|breakdown|analysis|"
    r"edit|amv|tribute|fan ?made|tiktok|#shorts|behind the scenes|bts|interview|featurette|"
    r"in \d+ minutes|summary|summarized|story of|ranking|ranked|top \d+|vs\.? |music video|soundtrack|"
    r"game ?play|lego|parody|spoof|honest)\b", re.I)
MIN_VIDEO_SECONDS, MAX_VIDEO_SECONDS = 15, 20 * 60
HANDLE_SECONDS = 3.0  # Extra footage around each approved segment, so lines can breathe and cuts don't clip action.
MIN_SEGMENT_SECONDS = 1.5
# Problems that do not make an official clip unusable: a corner channel logo is cropped out of the
# square frame, and end screens or title cards are cut by the closer look (run 19 lost 6 of 11 clips).
HARMLESS_PROBLEM = re.compile(r"(logo|watermark|\bbug\b|end[ _-]?screen|end[ _-]?card|title[ _-]?card|\bintro\b|\boutro\b|branding)", re.I)


SUBTITLE_PROBLEM = re.compile(r"subtitle|caption", re.I)
SUBTITLE_KEEP = 0.8  # Burned-in subtitles sit in the bottom of the frame; keep the top 80% of the picture.


def subtitled_only(result) -> bool:
    """Film footage whose only real fault is burned-in subtitles: a last resort, cropped above them
    (run 20 had no footage of the finger cut because its only upload was subtitled)."""
    if not result.same_film or not result.problems:
        return False
    return any(SUBTITLE_PROBLEM.search(p) for p in result.problems) and all(
        SUBTITLE_PROBLEM.search(p) or HARMLESS_PROBLEM.search(p) for p in result.problems) and not any(
        re.search(r"narrat|commentary|voice ?over|mirror|flipped|sped|speed|border|picture.in.picture", p, re.I)
        for p in result.problems)


def usable(result) -> bool:
    if not result.same_film:
        return False
    return result.clean or (bool(result.problems) and all(HARMLESS_PROBLEM.search(p) and not re.search(
        r"\b(subtitle|caption|narrat|commentary|voice ?over|mirror|flipped|sped|speed|border|picture.in.picture)",
        p, re.I) for p in result.problems))


# ------------------------------------------------------------------ schemas
class StoryItem(BaseModel):
    item_id: str = Field(description="Short id such as I1, I2 in story order")
    title: str = Field(description="A few words naming the moment, e.g. 'Car stolen at the gas station'")
    what_happens: str = Field(description="What is visibly on screen in this moment, concretely")
    story_context: str = Field("", description="Why this moment matters to the story: who wants what, what is at "
                                               "stake, what it leads to (for the narrator, from your film knowledge)")
    must_have: bool = Field(description="True if the story does not make sense without this moment")
    connects_to: list[str] = Field(default_factory=list, description="item_ids this moment leads to or explains")
    queries: list[str] = Field(description="3-6 broad, short YouTube searches people would actually title uploads with")


class StoryPlan(BaseModel):
    film: str
    year: int | None = None
    premise: str = Field("", description="2-3 sentences a newcomer needs first: who the hero is, what happened "
                                         "before, why he is in danger now")
    items: list[StoryItem]


class CandidatePick(BaseModel):
    video_id: str
    item_ids: list[str] = Field(description="Story items this upload probably contains")
    reason: str


class CandidateRanking(BaseModel):
    picks: list[CandidatePick]


class FoundSegment(BaseModel):
    item_id: str
    start: float = Field(description="Seconds from the start of this video")
    end: float
    shows: str = Field(description="What is actually visible in this segment")
    confidence: float = Field(ge=0, le=1)


class RefinedSegments(BaseModel):
    segments: list[FoundSegment] = Field(default_factory=list)


class Screening(BaseModel):
    same_film: bool = Field(description="The footage is from the requested film (not a sequel, remake or parody)")
    clean: bool = Field(description="No narration over the film, no burned-in subtitles or commentary text, no "
                                    "borders or picture-in-picture, not mirrored or sped up. A small channel logo in "
                                    "a corner and end screens are fine")
    problems: list[str] = Field(default_factory=list)
    segments: list[FoundSegment] = Field(default_factory=list)


# ------------------------------------------------------------------ YouTube I/O (replaceable in tests)
def netscape_cookies(text: str) -> str:
    """YT_COOKIES as a Netscape cookies.txt: accepts that format or a JSON export from a browser extension."""
    text = text.replace("\r\n", "\n").strip()
    if text.startswith("["):
        lines = []
        for c in json.loads(text):
            domain = c.get("domain", "")
            host_only = c.get("hostOnly", not domain.startswith("."))
            expires = int(c.get("expirationDate") or c.get("expires") or 0)
            lines.append("\t".join([domain, "FALSE" if host_only else "TRUE", c.get("path", "/"),
                                    "TRUE" if c.get("secure") else "FALSE", str(max(0, expires)),
                                    c.get("name", ""), c.get("value", "")]))
        text = "\n".join(lines)
    if not text.startswith("# Netscape HTTP Cookie File") and not text.startswith("# HTTP Cookie File"):
        text = "# Netscape HTTP Cookie File\n" + text
    return text + "\n"


_COOKIE_FILE: list[str] = []


def cookie_file() -> str | None:
    """Write the YT_COOKIES secret to a private temporary file once per process; never logged."""
    if _COOKIE_FILE:
        return _COOKIE_FILE[0]
    secret = os.environ.get("YT_COOKIES", "").strip()
    if not secret:
        return None
    handle, path = tempfile.mkstemp(prefix="yt-cookies-", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(netscape_cookies(secret))
    os.chmod(path, 0o600)
    atexit.register(lambda: os.path.exists(path) and os.remove(path))
    _COOKIE_FILE.append(path)
    return path


def _with_cookies(options: dict) -> dict:
    path = cookie_file()
    return {**options, "cookiefile": path} if path else options


def youtube_search(query: str, limit: int = 8) -> list[dict]:
    import yt_dlp

    with yt_dlp.YoutubeDL(_with_cookies({"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
                                         "skip_download": True, "socket_timeout": 30})) as ydl:
        result = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    found = []
    for entry in result.get("entries") or []:
        if entry and entry.get("id"):
            found.append({"video_id": entry["id"], "title": entry.get("title") or "",
                          "channel": entry.get("channel") or entry.get("uploader") or "",
                          "duration": float(entry.get("duration") or 0), "views": int(entry.get("view_count") or 0),
                          "url": f"https://www.youtube.com/watch?v={entry['id']}"})
    return found


def youtube_download_section(url: str, start: float, end: float, target: Path) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="segment-", dir=target.parent) as temp:
        options = {"outtmpl": str(Path(temp) / "part.%(ext)s"),
                   "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                   "merge_output_format": "mp4", "noplaylist": True, "quiet": True, "no_warnings": True,
                   "socket_timeout": 30, "retries": 2, "fragment_retries": 2, "cachedir": False,
                   "download_ranges": download_range_func(None, [(start, end)]), "force_keyframes_at_cuts": True,
                   "js_runtimes": {"deno": {}, "node": {}}}
        with yt_dlp.YoutubeDL(_with_cookies(options)) as ydl:
            ydl.download([url])
        produced = [p for p in Path(temp).glob("part.*") if p.suffix in {".mp4", ".mkv", ".webm"}]
        if not produced:
            raise RuntimeError("yt-dlp produced no file")
        produced[0].replace(target)
    return target


# ------------------------------------------------------------------ helpers
def film_name(request: dict) -> str:
    source = str(request.get("source", "")).strip()
    named = source.split(":", 1)[1].strip() if ":" in source else ""
    return named or str(request.get("movie") or request["title"]).strip()


def youtube_part(url: str, duration: float, fps: float = 0.5) -> dict:
    # Keyed as youtube_url (not file_uri) so the response cache tells videos apart.
    return {"youtube_url": url, "mime_type": "video/mp4", "fps": fps, "duration": duration}


def _merged_seconds(spans: list[tuple[float, float]]) -> float:
    total, last_end = 0.0, None
    for a, b in sorted(spans):
        if last_end is None or a > last_end:
            total += b - a
            last_end = b
        elif b > last_end:
            total += b - last_end
            last_end = b
    return total


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-1500:]}")


def _probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=120).stdout.strip()
    return float(out or 0)


def _has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120).stdout
    return bool(out.strip())


def detect_crop(path: Path) -> str | None:
    """Black bars baked into the picture (widescreen film clips): an ffmpeg crop filter, or None."""
    duration = _probe_duration(path)
    result = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{max(0.0, duration / 2 - 3):.2f}", "-t", "6",
                             "-i", str(path), "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                            capture_output=True, text=True, timeout=300)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=120).stdout.strip()
    if not found or "," not in size:
        return None
    w, h, x, y = map(int, found[-1])
    width, height = map(int, size.split(",")[:2])
    # Only real bars: a clearly smaller picture that is still most of the frame.
    if (h < height * 0.95 or w < width * 0.95) and h > height * 0.4 and w > width * 0.4:
        return f"crop={w}:{h}:{x}:{y}"
    return None


def normalize_segment(source: Path, target: Path, *, subtitle_band: bool = False) -> Path:
    """One format for every pooled clip: black bars removed, 1920x1080, 30 fps, H.264 + 48 kHz stereo AAC.
    With subtitle_band, the bottom of the picture (burned-in subtitles) is cropped away first."""
    crop = detect_crop(source)
    band = f"crop=iw:trunc(ih*{SUBTITLE_KEEP}/2)*2:0:0," if subtitle_band else ""
    video = ((crop + "," if crop else "") + band + "scale=1920:1080:force_original_aspect_ratio=increase,"
             "crop=1920:1080,setsar=1,fps=30,format=yuv420p")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if _has_audio(source):
        args += ["-map", "0:v:0", "-map", "0:a:0", "-af", "aresample=48000,aformat=channel_layouts=stereo"]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    _run(args + ["-vf", video, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                 "-movflags", "+faststart", str(target)])
    return target


# ------------------------------------------------------------------ prompts
PLAN_SYSTEM = ("You plan the footage for a narrated movie recap Short. You know the film well, but you only name "
               "moments that really happen in it. Return JSON matching the schema.")
RANK_SYSTEM = ("You choose which YouTube uploads to watch for a movie recap. Prefer official studio clip channels "
               "and plain movie footage at good resolution. Avoid recaps, reactions, reviews, trailers, edits, "
               "compilations with commentary and reuploads with borders or logos. Return JSON matching the schema.")
SCREEN_SYSTEM = ("You check a YouTube video for usable recap footage. Watch it and report only what is visible. "
                 "Times are seconds from the start of this video. Return JSON matching the schema.")


def plan_story(client: GeminiClient, request: dict, film: str) -> StoryPlan:
    target = float(request.get("target_seconds", 100))
    prompt = (f"Film: {film}\nShort title: {request['title']}\nBrief: {request.get('brief', '')}\n"
              f"Notes: {request.get('story_notes', '')}\nTarget narration: about {target:.0f} seconds.\n\n"
              "Plan a recap that someone who never saw the film can follow from start to end. Write the premise "
              "(who the hero is, what happened before, why he is in danger). Then list 6-10 story items in the order "
              "they happen in the film:\n"
              "- first the moment that sets up the story (the inciting event, e.g. the bounty being placed), then "
              "the film's most famous or action-packed moments, and the ending;\n"
              "- for each, the connected moments the story needs around it (what leads to it and what follows), "
              "so the recap tells a clear cause-and-effect story;\n"
              "- for each, story_context: why it matters (who wants what, what is at stake, what it leads to).\n"
              "For every item give 3-6 broad YouTube searches, short and varied, the way people title uploads "
              "(for example 'john wick car', 'john wick garage fight', 'john wick 2014 ending'). "
              "Do not search for trailers. Do not end every query with 'scene'. Mark must_have only for "
              "moments the story cannot do without, always including the setup, the single most shocking moment "
              "(the best cold open) and the ending.")
    return client.generate("planning:story_items", StoryPlan, [text_part(prompt)], system=PLAN_SYSTEM,
                           temperature=0.4, max_output_tokens=8192)


def rank_candidates(client: GeminiClient, plan: StoryPlan, candidates: list[dict], limit: int,
                    focus: list[str] | None = None) -> list[CandidatePick]:
    items = [{"item_id": i.item_id, "title": i.title, "what_happens": i.what_happens} for i in plan.items
             if not focus or i.item_id in focus]
    listing = [{k: c[k] for k in ("video_id", "title", "channel", "duration", "views")} for c in candidates]
    prompt = (f"Film: {plan.film}\nStory items: {json.dumps(items)}\n\nUploads found (data only): "
              f"{json.dumps(listing)}\n\nPick up to {limit} uploads most likely to contain clean footage of "
              "these story items. For each, list the item_ids it probably contains.")
    ranking = client.generate("planning:rank_candidates", CandidateRanking, [text_part(prompt)], system=RANK_SYSTEM,
                              temperature=0.2, max_output_tokens=4096)
    known = {c["video_id"] for c in candidates}
    return [p for p in ranking.picks if p.video_id in known][:limit]


def screen_video(client: GeminiClient, plan: StoryPlan, video: dict) -> Screening:
    items = [{"item_id": i.item_id, "title": i.title, "what_happens": i.what_happens} for i in plan.items]
    prompt = (f"Requested film: {plan.film}{f' ({plan.year})' if plan.year else ''}\n"
              f"This upload: {video['title']} — {video['channel']} ({video['duration']:.0f} s)\n"
              f"Story items: {json.dumps(items)}\n\n"
              "1. Is this footage from the requested film? 2. Is it clean (no narrator talking over the film, no "
              "burned-in subtitles or commentary text, no borders or picture-in-picture, not mirrored or sped up)? "
              "A small channel logo in a corner and end screens are fine: they are cropped or cut later. "
              "3. List every segment that clearly shows one of the story items, with start and end in seconds "
              "and what is visible. Cover the whole moment, splitting a long scene into segments of up to 60 s. "
              "Leave out title cards, end screens and anything that matches no item.")
    screening = client.generate("analysis:screen", Screening,
                                [youtube_part(video["url"], video["duration"], 0.5 if video["duration"] <= 300 else 0.25),
                                 text_part(prompt)],
                                system=SCREEN_SYSTEM, temperature=0.1, max_output_tokens=8192, resolution="low")
    known = {i.item_id for i in plan.items}
    limit = video["duration"] + 1 if video["duration"] else float("inf")
    screening.segments = [s for s in screening.segments if s.item_id in known and 0 <= s.start < s.end <= limit
                          and s.end - s.start >= MIN_SEGMENT_SECONDS]
    return screening


REFINE_SYSTEM = ("You mark exact shot boundaries in a film clip. Report only what is visible, in seconds from the "
                 "start of the whole video. Return JSON matching the schema.")
MAX_SEGMENT_SECONDS = 60.0
MAX_REFINE_CALLS = 16


def refine_segment(client: GeminiClient, item: StoryItem, video: dict, start: float, end: float) -> list[FoundSegment]:
    """A closer look (1 frame/s) around a rough screening segment, for tight start and end times."""
    duration = video["duration"] or end + 15
    a, b = max(0.0, start - 25), min(duration, end + 25)
    part = {**youtube_part(video["url"], duration, 1.0), "start": round(a, 3), "end": round(b, 3)}
    prompt = (f"Watch {a:.0f}-{b:.0f} s of this video and find the exact moments that show: {item.title} — "
              f"{item.what_happens}\nReturn 1-3 continuous stretches of 3-{MAX_SEGMENT_SECONDS:.0f} s that together cover the "
              "whole moment (its build-up, the action and its result), start and end in seconds from the start of "
              "the whole video. Leave out title cards, end screens and unrelated shots. Return no segments if the "
              "moment is not there.")
    result = client.generate("analysis:refine", RefinedSegments, [part, text_part(prompt)], system=REFINE_SYSTEM,
                             temperature=0.1, max_output_tokens=4096, resolution="low")
    found = [s for s in result.segments if s.end > s.start]
    # Some answers count from the start of the watched window; shift those into the video's time.
    if found and all(s.end <= (b - a) + 0.5 for s in found) and not all(a - 0.5 <= s.start and s.end <= b + 0.5 for s in found):
        for s in found:
            s.start, s.end = s.start + a, s.end + a
    tight = []
    for s in found:
        s.item_id = item.item_id
        s.start, s.end = max(a, s.start), min(b, s.end)
        if s.end - s.start > MAX_SEGMENT_SECONDS:
            middle = (s.start + s.end) / 2
            s.start, s.end = middle - MAX_SEGMENT_SECONDS / 2, middle + MAX_SEGMENT_SECONDS / 2
        if s.end - s.start >= MIN_SEGMENT_SECONDS:
            tight.append(s)
    return tight


def handle_range(c: dict, chosen: list[dict], duration: float) -> tuple[float, float]:
    """The segment plus its handles, never reaching into another chosen segment of the same video:
    where two are close, the gap between them is split, so the pool never shows a moment twice."""
    a, b = max(0.0, c["start"] - HANDLE_SECONDS), min(duration, c["end"] + HANDLE_SECONDS)
    for o in chosen:
        if o is c or o["video_id"] != c["video_id"]:
            continue
        if o["end"] <= c["start"]:
            a = max(a, (o["end"] + c["start"]) / 2 if c["start"] - o["end"] < 2 * HANDLE_SECONDS else a)
        elif o["start"] >= c["end"]:
            b = min(b, (c["end"] + o["start"]) / 2 if o["start"] - c["end"] < 2 * HANDLE_SECONDS else b)
    return round(a, 3), round(b, 3)


# ------------------------------------------------------------------ main entry
def discover_footage(request: dict, workdir: Path, outdir: Path, *, transport=None,
                     search: Callable = youtube_search, download: Callable = youtube_download_section,
                     log: Callable = print, sleep=None, max_screen: int = 12) -> tuple[Path, str]:
    """Return (pooled source video, story notes describing the pool in story order)."""
    film = film_name(request)
    key = hashlib.sha256(json.dumps([film, request["title"], request.get("brief", "")]).encode()).hexdigest()[:16]
    base = Path(workdir) / "discovery" / key
    base.mkdir(parents=True, exist_ok=True)
    outdir = Path(outdir)
    settings = settings_from_request(request)
    ledger = CostLedger(settings.max_budget_usd, outdir / "discovery-cost.json", free_tier=settings.free_tier)
    client = GeminiClient(settings, ledger, Path(workdir) / "gemini" / "cache", transport, sleep=sleep)
    client.resolve_models()
    report: dict = {"film": film, "started_epoch": round(time.time(), 1), "searches": [], "screened": []}

    def save():
        _write_json(outdir / "discovery-report.json", report)

    log(f"Discovery: planning story items for {film}")
    plan = plan_story(client, request, film)
    report["story_plan"] = plan.model_dump()
    _write_json(base / "story-plan.json", plan.model_dump())  # The storyline path writes from this plan.
    # Screening and refine answers name this plan's item ids, so their caches are per plan.
    plan_key = hashlib.sha256(plan.model_dump_json().encode()).hexdigest()[:10]
    save()

    candidates: dict[str, dict] = {}
    searched: set[str] = set()

    def run_searches(queries: list[str]):
        for query in queries:
            if query.lower() in searched:
                continue
            searched.add(query.lower())
            try:
                results = search(query)
            except Exception as exc:  # noqa: BLE001 - one failed search is not fatal
                report["searches"].append({"query": query, "error": str(exc)[:300]})
                continue
            kept = [r for r in results if MIN_VIDEO_SECONDS <= r["duration"] <= MAX_VIDEO_SECONDS
                    and not REJECT_TITLE.search(r["title"])]
            report["searches"].append({"query": query, "found": len(results), "kept": [r["video_id"] for r in kept]})
            for r in kept:
                candidates.setdefault(r["video_id"], r)

    run_searches([q for item in plan.items for q in item.queries[:2]])
    if not candidates:
        save()
        raise RuntimeError("Footage discovery found no usable YouTube uploads (searches failed or every result was a "
                           "trailer/recap/edit). See discovery-report.json. Supply a source video instead.")

    screenings: dict[str, Screening] = {}

    def screen(picks: list[CandidatePick]):
        for pick in picks:
            if pick.video_id in screenings:
                continue
            video = candidates[pick.video_id]
            cache = base / "screen" / f"{plan_key}-{pick.video_id}.json"
            if cache.is_file():
                result = Screening.model_validate_json(cache.read_text())
            else:
                log(f"Discovery: Gemini is watching {video['title'][:70]}")
                try:
                    result = screen_video(client, plan, video)
                except Exception as exc:  # noqa: BLE001 - a video Gemini cannot watch is skipped
                    report["screened"].append({"video_id": pick.video_id, "title": video["title"],
                                               "error": str(exc)[:300]})
                    continue
                _write_json(cache, result.model_dump())
            screenings[pick.video_id] = result
            report["screened"].append({"video_id": pick.video_id, "title": video["title"], "channel": video["channel"],
                                       "url": video["url"], "same_film": result.same_film, "clean": result.clean,
                                       "usable": usable(result),
                                       "problems": result.problems, "segments": [s.model_dump() for s in result.segments]})
            save()

    screen(rank_candidates(client, plan, list(candidates.values()), max_screen))

    def coverage() -> dict[str, list[tuple[str, FoundSegment]]]:
        found: dict[str, list[tuple[str, FoundSegment]]] = {i.item_id: [] for i in plan.items}
        for video_id, result in screenings.items():
            if usable(result):
                for seg in result.segments:
                    found[seg.item_id].append((video_id, seg))
        # Last resort for a moment with no clean footage: subtitled film footage, cropped above the subtitles.
        clean = {k for k, v in found.items() if v}
        for video_id, result in screenings.items():
            if subtitled_only(result):
                for seg in result.segments:
                    if seg.item_id not in clean:
                        found[seg.item_id].append((video_id, seg))
        return found

    def subtitled(video_id: str) -> bool:
        return not usable(screenings[video_id]) and subtitled_only(screenings[video_id])

    # Every moment without clean footage gets one more search round (must-have first, at most four).
    uncovered = [i for i in plan.items if not any(not subtitled(v) for v, _ in coverage()[i.item_id])]
    missing = sorted(uncovered, key=lambda i: not i.must_have)[:4]
    if missing:
        log(f"Discovery: searching again for {', '.join(i.title for i in missing)}")
        run_searches([q for item in missing for q in item.queries[2:]])
        fresh = [c for vid, c in candidates.items() if vid not in screenings]
        if fresh:
            screen(rank_candidates(client, plan, fresh, 4, focus=[i.item_id for i in missing]))

    found = coverage()
    target = float(request.get("target_seconds", 100))
    # Footage per moment from the moments that actually have footage: a 100 s Short over few
    # covered moments still needs about 2.6x its length in usable footage (run 19 pooled 122 s).
    covered = sum(1 for i in plan.items if found[i.item_id])
    per_item = min(90.0, max(25.0, 2.6 * target / max(1, covered)))
    per_item_segments = 4
    chosen: list[dict] = []
    for item in plan.items:
        spans: list[tuple[float, float]] = []
        for video_id, seg in sorted(found[item.item_id], key=lambda vs: -vs[1].confidence):
            if _merged_seconds(spans) >= per_item or len(spans) >= per_item_segments:
                break
            if any(c["video_id"] == video_id and seg.start < c["end"] and seg.end > c["start"] for c in chosen):
                continue
            chosen.append({"item_id": item.item_id, "item_title": item.title, "video_id": video_id,
                           "url": candidates[video_id]["url"], "title": candidates[video_id]["title"],
                           "start": seg.start, "end": seg.end, "shows": seg.shows, "confidence": seg.confidence,
                           "subtitled": subtitled(video_id)})
            spans.append((seg.start, seg.end))
    # Closer look at each chosen stretch: screening at a frame every 2-4 s gives rough, often whole-clip spans.
    items = {i.item_id: i for i in plan.items}
    refined: list[dict] = []
    refine_calls = 0
    for c in chosen:
        cache = base / "refine" / f"{plan_key}-{c['video_id']}-{c['item_id']}-{c['start']:.1f}-{c['end']:.1f}.json"
        try:
            if cache.is_file():
                tight = RefinedSegments.model_validate_json(cache.read_text()).segments
            elif refine_calls >= MAX_REFINE_CALLS:
                tight = []  # Request budget: keep the rough span, capped below.
            else:
                refine_calls += 1
                log(f"Discovery: marking exact times for {c['item_title']} in {c['video_id']}")
                tight = refine_segment(client, items[c["item_id"]], candidates[c["video_id"]], c["start"], c["end"])
                _write_json(cache, RefinedSegments(segments=tight).model_dump())
        except Exception as exc:  # noqa: BLE001 - keep the rough span, capped
            c["refine_error"] = str(exc)[:200]
            tight = []
        if tight:
            refined += [{**c, "start": s.start, "end": s.end, "shows": s.shows, "confidence": s.confidence,
                         "rough": [c["start"], c["end"]]} for s in tight]
        else:
            middle = (c["start"] + c["end"]) / 2
            half = min(MAX_SEGMENT_SECONDS, c["end"] - c["start"]) / 2
            refined.append({**c, "start": middle - half, "end": middle + half, "rough": [c["start"], c["end"]]})
    chosen = []
    for item in plan.items:
        total = 0.0
        taken = 0
        for c in sorted((r for r in refined if r["item_id"] == item.item_id), key=lambda r: -r["confidence"]):
            if total >= per_item or taken >= per_item_segments + 1:
                break
            if any(o["video_id"] == c["video_id"] and c["start"] < o["end"] and c["end"] > o["start"] for o in chosen):
                continue
            chosen.append(c)
            taken += 1
            total += c["end"] - c["start"]
    chosen.sort(key=lambda c: ([i.item_id for i in plan.items].index(c["item_id"]), c["video_id"], c["start"]))
    report["coverage"] = [{"item_id": i.item_id, "title": i.title, "must_have": i.must_have,
                           "seconds": round(sum(c["end"] - c["start"] for c in chosen if c["item_id"] == i.item_id), 1)}
                          for i in plan.items]
    report["missing_items"] = [c["title"] for c in report["coverage"] if c["seconds"] == 0]
    save()
    if not chosen:
        raise RuntimeError("Gemini found no clean footage of the story items in the YouTube uploads it watched. "
                           "See discovery-report.json for every verdict. Supply a source video instead.")

    clips_dir, pooled = base / "segments", []
    clips_dir.mkdir(parents=True, exist_ok=True)
    for index, c in enumerate(chosen):
        a, b = handle_range(c, chosen, candidates[c["video_id"]]["duration"] or (c["end"] + HANDLE_SECONDS))
        raw = clips_dir / f"{c['video_id']}-{a:.1f}-{b:.1f}.mp4"
        norm = clips_dir / f"{c['video_id']}-{a:.1f}-{b:.1f}-{'sub' if c.get('subtitled') else 'norm'}.mp4"
        try:
            if not norm.is_file():
                if not raw.is_file():
                    log(f"Discovery: downloading {c['item_title']} ({a:.0f}-{b:.0f}s of {c['video_id']})")
                    download(c["url"], a, b, raw)
                normalize_segment(raw, norm, subtitle_band=bool(c.get("subtitled")))
            c.update(downloaded_start=a, downloaded_end=b, file=str(norm))
            pooled.append(c)
        except Exception as exc:  # noqa: BLE001 - recorded; other segments may still work
            c["download_error"] = str(exc)[:300]
    report["chosen"] = chosen
    save()
    if not pooled:
        links = "; ".join(f"{c['url']}&t={int(c['start'])} ({c['start']:.0f}-{c['end']:.0f}s, {c['item_title']})"
                          for c in chosen[:8])
        raise RuntimeError("Gemini approved footage but YouTube blocked every download from this machine. Download "
                           f"these segments and supply them as the source instead: {links}")

    pool = base / "pool.mp4"
    listing = base / "pool.txt"
    listing.write_text("".join(f"file '{p['file']}'\n" for p in pooled))
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
          "-c", "copy", "-movflags", "+faststart", str(pool)])
    offset, pool_map, notes = 0.0, [], []
    for p in pooled:
        length = _probe_duration(Path(p["file"]))
        pool_map.append({"pool_start": round(offset, 3), "pool_end": round(offset + length, 3), "item_id": p["item_id"],
                         "item_title": p["item_title"], "video_id": p["video_id"], "url": p["url"],
                         "source_start": p["downloaded_start"], "source_end": p["downloaded_end"], "shows": p["shows"]})
        why = items[p["item_id"]].story_context
        notes.append(f"[{offset:.0f}-{offset + length:.0f}s] {p['item_title']}: shows {p['shows']}"
                     + (f". Why it matters: {why}" if why else ""))
        offset += length
    _write_json(base / "pool-map.json", pool_map)
    report["pool"] = {"seconds": round(offset, 1), "segments": pool_map}
    report["finished_epoch"] = round(time.time(), 1)
    save()
    ledger.save()
    log(f"Discovery: footage pool ready, {offset:.0f}s from {len({p['video_id'] for p in pooled})} upload(s)")
    story = (f"Film: {plan.film}. Premise: {plan.premise}\nThe footage is a pool of film clips joined in story "
             "order:\n" + "\n".join(notes))
    return pool, story
