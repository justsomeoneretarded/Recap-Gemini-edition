"""Footage discovery: story plan -> broad search -> Gemini screens links -> only approved segments download."""
import json
import subprocess
from pathlib import Path

import pytest

from shorts_pipeline.gemini import discovery
from gemini_fakes import FakeAPIError, make_source

PLAN = {"film": "John Wick", "year": 2014, "items": [
    {"item_id": "I1", "title": "Car stolen", "what_happens": "Thugs take John's Mustang", "must_have": True,
     "connects_to": ["I2"], "queries": ["john wick car", "john wick mustang", "john wick gas station"]},
    {"item_id": "I2", "title": "Digging up the floor", "what_happens": "John smashes the basement floor", "must_have": True,
     "connects_to": [], "queries": ["john wick floor", "john wick hammer", "john wick basement"]},
    {"item_id": "I3", "title": "Nightclub fight", "what_happens": "John fights through the club", "must_have": False,
     "connects_to": [], "queries": ["john wick club fight"]}]}

UPLOADS = {
    "john wick car": [{"video_id": "carA", "title": "John Wick (2014) - Car Theft", "channel": "Movieclips", "duration": 40},
                      {"video_id": "trl", "title": "John Wick Official Trailer", "channel": "Lionsgate", "duration": 150},
                      {"video_id": "rcp", "title": "John Wick Recap in 10 minutes", "channel": "X", "duration": 600}],
    "john wick floor": [{"video_id": "capt", "title": "john wick floor", "channel": "fan", "duration": 30}],
    "john wick basement": [{"video_id": "floorB", "title": "John Wick - I'm back", "channel": "Movieclips", "duration": 35}],
}


class Transport:
    def __init__(self):
        self.calls = []

    def classify(self, exc):
        return getattr(exc, "category", "request"), str(exc)

    def list_models(self):
        return ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

    def generate(self, model, parts, *, system, schema, **config):
        self.calls.append((schema.__name__, parts))
        text = " ".join(p.get("text", "") for p in parts)
        if schema.__name__ == "StoryPlan":
            return self._ok(PLAN)
        if schema.__name__ == "RefinedSegments":  # The closer look tightens each rough span.
            url = next(p["youtube_url"] for p in parts if "youtube_url" in p)
            tight = {"carA": (6, 14), "floorB": (11, 20)}[url.rsplit("=", 1)[1]]
            return self._ok({"segments": [{"item_id": "?", "start": tight[0], "end": tight[1], "shows": "tight",
                                           "confidence": 0.95}]})
        if schema.__name__ == "CandidateRanking":
            listed = json.loads(text.split("Uploads found (data only): ")[1].split("\n\n")[0])
            return self._ok({"picks": [{"video_id": u["video_id"], "item_ids": ["I1"], "reason": "r"} for u in listed]})
        url = next(p["youtube_url"] for p in parts if "youtube_url" in p)
        if url.endswith("carA"):
            return self._ok({"same_film": True, "clean": True, "segments": [
                {"item_id": "I1", "start": 5, "end": 15, "shows": "Mustang driven off", "confidence": 0.9},
                {"item_id": "I9", "start": 1, "end": 4, "shows": "unknown item", "confidence": 0.9}]})
        if url.endswith("capt"):  # burned-in captions: rejected even though it shows the item
            return self._ok({"same_film": True, "clean": False, "problems": ["burned-in captions"], "segments": [
                {"item_id": "I2", "start": 2, "end": 9, "shows": "floor", "confidence": 0.9}]})
        return self._ok({"same_film": True, "clean": True, "segments": [
            {"item_id": "I2", "start": 10, "end": 22, "shows": "John smashes the floor with a hammer", "confidence": 0.8}]})

    @staticmethod
    def _ok(data):
        return {"text": json.dumps(data), "finish_reason": "STOP", "prompt_tokens": 100, "cached_tokens": 0,
                "output_tokens": 50, "thinking_tokens": 0, "audio_tokens": 0}


def search(query, limit=8):
    return [{**u, "views": 1000, "url": f"https://www.youtube.com/watch?v={u['video_id']}"}
            for u in UPLOADS.get(query, [])]


def request(**extra):
    return {"title": "Everyone Wants Him Dead", "source": "discover:John Wick (2014)", "brief": "b", "story_notes": "",
            "target_seconds": 60, "gemini_preset": "free", **extra}


def test_discovery_screens_links_downloads_only_approved_segments_and_pools_them(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    master = make_source(tmp_path / "master.mp4", 40)
    downloads = []

    def download(url, start, end, target):
        downloads.append((url.rsplit("=", 1)[1], start, end))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(master),
                        "-c:v", "libx264", "-c:a", "aac", str(target)], check=True)
        return target

    transport = Transport()
    pool, notes = discovery.discover_footage(request(), tmp_path / "work", tmp_path / "out", transport=transport,
                                             search=search, download=download, log=lambda *_: None,
                                             sleep=lambda s: None)
    report = json.loads((tmp_path / "out" / "discovery-report.json").read_text())
    assert report["film"] == "John Wick (2014)"
    # Trailers and recaps never reach Gemini; the captioned upload is watched but rejected.
    watched = [s["video_id"] for s in report["screened"]]
    assert "trl" not in watched and "rcp" not in watched and "capt" in watched
    # Only the approved segments (with 3 s handles) were downloaded; nothing from the rejected upload.
    assert sorted(d[0] for d in downloads) == ["carA", "floorB"]
    assert ("carA", 3.0, 17.0) in downloads  # Refined 6-14 s plus 3 s handles.
    assert {c["item_id"]: c["seconds"] for c in report["coverage"]} == {"I1": 8.0, "I2": 9.0, "I3": 0}
    assert report["missing_items"] == ["Nightclub fight"]
    # The pool is in story order and the notes tell the pipeline what each part shows.
    pool_map = report["pool"]["segments"]
    assert [p["item_id"] for p in pool_map] == ["I1", "I2"] and pool.is_file()
    assert abs(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(pool)], capture_output=True, text=True).stdout) - 29) < 1.0  # 14 s + 15 s with handles
    assert "Car stolen" in notes and "Digging up the floor" in notes
    # A rerun keeps the found pool: search results change from day to day (run 24 lost its opening footage
    # and the shot labels and script built on it), so nothing is searched, screened or downloaded again.
    calls = len(transport.calls)
    again, again_notes = discovery.discover_footage(
        request(), tmp_path / "work", tmp_path / "out2", transport=transport,
        search=lambda *a, **k: pytest.fail("searched again"), download=lambda *a: pytest.fail("downloaded again"),
        log=lambda *_: None, sleep=lambda s: None)
    assert len(transport.calls) == calls and again == pool and again_notes == notes
    assert json.loads((tmp_path / "out2" / "discovery-report.json").read_text())["reused_pool"] is True
    # rediscover=true searches again (every Gemini answer and download is still cached).
    searched = []
    discovery.discover_footage(request(rediscover=True), tmp_path / "work", tmp_path / "out3", transport=transport,
                               search=lambda q, limit=8: searched.append(q) or search(q, limit),
                               download=lambda *a: pytest.fail("downloaded again"), log=lambda *_: None,
                               sleep=lambda s: None)
    assert searched and len(transport.calls) == calls


def test_must_have_items_without_footage_get_a_second_search_round(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    seen = []

    def tracking(query, limit=8):
        seen.append(query)
        return search(query, limit)

    with pytest.raises(RuntimeError, match="blocked every download"):
        discovery.discover_footage(request(), tmp_path / "w", tmp_path / "o", transport=Transport(), search=tracking,
                                   download=lambda *a: (_ for _ in ()).throw(RuntimeError("Sign in to confirm")),
                                   log=lambda *_: None, sleep=lambda s: None)
    # First round uses two queries per item; I2 ("floor" upload is captioned) triggers its third query.
    assert seen[:5] == ["john wick car", "john wick mustang", "john wick floor", "john wick hammer", "john wick club fight"]
    assert "john wick basement" in seen
    report = json.loads((tmp_path / "o" / "discovery-report.json").read_text())
    assert all("download_error" in c for c in report["chosen"])


def test_youtube_part_keeps_videos_apart_in_the_cache_and_names_film():
    from shorts_pipeline.gemini.client import GeminiClient

    a, b = discovery.youtube_part("https://youtu.be/a", 30), discovery.youtube_part("https://youtu.be/b", 30)
    assert GeminiClient._stable([a]) != GeminiClient._stable([b])
    assert discovery.film_name({"source": "discover", "title": "Wick"}) == "Wick"
    assert discovery.film_name({"source": "discover: John Wick (2014)", "title": "x"}) == "John Wick (2014)"
    assert discovery.REJECT_TITLE.search("John Wick Official Trailer") and not discovery.REJECT_TITLE.search("John Wick - Car")


def test_cookies_accept_netscape_or_json_exports_and_stay_private(tmp_path, monkeypatch):
    netscape = discovery.netscape_cookies(".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tabc")
    assert netscape.startswith("# Netscape HTTP Cookie File\n") and "\tSID\tabc" in netscape
    exported = json.dumps([{"domain": ".youtube.com", "hostOnly": False, "path": "/", "secure": True,
                            "expirationDate": 1999999999.5, "name": "SID", "value": "abc"}])
    assert discovery.netscape_cookies(exported).splitlines()[1] == ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tabc"
    monkeypatch.setattr(discovery, "_COOKIE_FILE", [])
    monkeypatch.setenv("YT_COOKIES", exported)
    path = discovery.cookie_file()
    assert oct(Path(path).stat().st_mode & 0o777) == "0o600"
    assert discovery._with_cookies({"quiet": True})["cookiefile"] == path


def test_refine_shifts_window_relative_times_and_caps_long_spans(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    from shorts_pipeline.gemini.client import CostLedger, GeminiClient
    from shorts_pipeline.gemini.settings import settings_from_request

    class Window(Transport):
        def generate(self, model, parts, *, system, schema, **config):
            # Answers counted from the start of the 75-155 s window, one span far too long.
            return self._ok({"segments": [{"item_id": "x", "start": 5, "end": 12, "shows": "s", "confidence": 0.9},
                                          {"item_id": "x", "start": 10, "end": 80, "shows": "l", "confidence": 0.7}]})

    client = GeminiClient(settings_from_request({"gemini_preset": "free"}), CostLedger(5.0, None), tmp_path,
                          Window(), sleep=lambda s: None)
    item = discovery.StoryItem(item_id="I1", title="t", what_happens="w", must_have=True, queries=["q"])
    found = discovery.refine_segment(client, item, {"url": "https://youtu.be/v", "duration": 300}, 100, 130)
    assert [(s.item_id, s.start, s.end) for s in found] == [("I1", 80, 87), ("I1", 90, 150)]  # 70 s capped to 60 s


def test_black_bars_are_cropped_from_widescreen_clips(tmp_path):
    letterboxed = tmp_path / "wide.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x274:rate=24:duration=8",
                    "-vf", "pad=640:360:0:43:black", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(letterboxed)],
                   check=True)
    crop = discovery.detect_crop(letterboxed)
    assert crop and crop.startswith("crop=640:27")
    assert discovery.detect_crop(make_source(tmp_path / "full.mp4", 8)) is None


def test_a_corner_logo_or_end_screen_does_not_reject_an_official_clip():
    def screening(clean, problems, same=True):
        return discovery.Screening(same_film=same, clean=clean, problems=problems)
    assert discovery.usable(screening(False, ["The video contains a persistent 'B' logo watermark in the top left corner."]))
    assert discovery.usable(screening(False, ["watermark", "end_screen"]))
    assert not discovery.usable(screening(False, ["burned-in subtitles"]))
    assert not discovery.usable(screening(False, ["watermark", "narration over the film"]))
    assert not discovery.usable(screening(False, []))
    assert not discovery.usable(screening(True, [], same=False))


def test_subtitled_footage_is_a_cropped_last_resort_for_an_uncovered_moment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123456")
    master = make_source(tmp_path / "master.mp4", 40)
    downloads = []

    class Subtitled(Transport):
        def generate(self, model, parts, *, system, schema, **config):
            url = next((p["youtube_url"] for p in parts if "youtube_url" in p), "")
            if schema.__name__ == "Screening" and url.endswith("clubS"):
                self.calls.append((schema.__name__, parts))
                return self._ok({"same_film": True, "clean": False, "problems": ["burned_in_subtitles"], "segments": [
                    {"item_id": "I3", "start": 4, "end": 16, "shows": "club fight", "confidence": 0.8}]})
            if schema.__name__ == "RefinedSegments" and url.endswith("clubS"):
                return self._ok({"segments": []})
            return super().generate(model, parts, system=system, schema=schema, **config)

    def subtitled_search(query, limit=8):
        if query == "john wick club fight":
            return [{"video_id": "clubS", "title": "John Wick - Club Fight", "channel": "Clips", "duration": 30,
                     "views": 10, "url": "https://www.youtube.com/watch?v=clubS"}]
        return search(query, limit)

    def download(url, start, end, target):
        downloads.append(url.rsplit("=", 1)[1])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(master),
                        "-c:v", "libx264", "-c:a", "aac", str(target)], check=True)
        return target

    discovery.discover_footage(request(), tmp_path / "work", tmp_path / "out", transport=Subtitled(),
                               search=subtitled_search, download=download, log=lambda *_: None, sleep=lambda s: None)
    report = json.loads((tmp_path / "out" / "discovery-report.json").read_text())
    club = [c for c in report["chosen"] if c["video_id"] == "clubS"]
    # Only the moment with no clean footage takes the subtitled upload, normalized above the subtitles.
    assert club and all(c["item_id"] == "I3" and c["subtitled"] and c["file"].endswith("-sub.mp4") for c in club)
    assert not any(c["subtitled"] for c in report["chosen"] if c["video_id"] != "clubS")
    assert {c["item_id"]: c["seconds"] for c in report["coverage"]}["I3"] > 0
    assert discovery.subtitled_only(discovery.Screening(same_film=True, clean=False, problems=["burned_in_subtitles"]))
    assert not discovery.subtitled_only(discovery.Screening(same_film=True, clean=False,
                                                            problems=["burned-in subtitles", "narration over the film"]))
