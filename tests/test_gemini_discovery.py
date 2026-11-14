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
    # A rerun reuses every Gemini answer and downloaded segment.
    calls = len(transport.calls)
    discovery.discover_footage(request(), tmp_path / "work", tmp_path / "out2", transport=transport, search=search,
                               download=lambda *a: pytest.fail("downloaded again"), log=lambda *_: None,
                               sleep=lambda s: None)
    assert len(transport.calls) == calls


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
            # Answers counted from the start of the 85-145 s window, one span far too long.
            return self._ok({"segments": [{"item_id": "x", "start": 5, "end": 12, "shows": "s", "confidence": 0.9},
                                          {"item_id": "x", "start": 10, "end": 58, "shows": "l", "confidence": 0.7}]})

    client = GeminiClient(settings_from_request({"gemini_preset": "free"}), CostLedger(5.0, None), tmp_path,
                          Window(), sleep=lambda s: None)
    item = discovery.StoryItem(item_id="I1", title="t", what_happens="w", must_have=True, queries=["q"])
    found = discovery.refine_segment(client, item, {"url": "https://youtu.be/v", "duration": 300}, 100, 130)
    assert [(s.item_id, s.start, s.end) for s in found] == [("I1", 90, 97), ("I1", 104, 134)]
