"""Optional real-API checks. They never run in ordinary CI.

Run explicitly, with your key (the free tier is fine; a few requests are used):
    RUN_GEMINI_LIVE=1 GEMINI_API_KEY=... python -m pytest tests/test_gemini_live.py -v
The 'Gemini Live Check' workflow does the same from GitHub Actions on demand.
"""
import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(not (os.getenv("RUN_GEMINI_LIVE") == "1" and os.getenv("GEMINI_API_KEY")),
                                reason="live Gemini tests run only with RUN_GEMINI_LIVE=1 and GEMINI_API_KEY")


def _client(tmp_path, preset="free"):
    from shorts_pipeline.gemini.client import CostLedger, GeminiClient
    from shorts_pipeline.gemini.settings import settings_from_request
    settings = settings_from_request({"gemini_preset": preset, "max_budget_usd": 0.25})
    return GeminiClient(settings, CostLedger(0.25, tmp_path / "cost.json", free_tier=settings.free_tier),
                        tmp_path / "cache")


def test_live_models_resolve_for_every_preset(tmp_path):
    for preset in ("free", "economy", "balanced", "quality"):
        try:
            resolved = _client(tmp_path, preset).resolve_models()
        except Exception as exc:  # Paid presets may list no usable model for a free key.
            print(preset, "unavailable:", exc)
            assert preset != "free", exc
            continue
        assert set(resolved) == {"analysis", "planning", "writing", "matching", "review"}, preset
        print(preset, resolved)


def test_live_structured_video_analysis_of_a_synthetic_clip(tmp_path):
    from shorts_pipeline.gemini.analysis import ANALYSIS_SYSTEM, make_proxy
    from shorts_pipeline.gemini.client import text_part, video_part
    from shorts_pipeline.gemini.schemas import WindowAnalysis
    source = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=6",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    client = _client(tmp_path)
    client.resolve_models()
    proxy, meta = make_proxy(source, tmp_path)
    file = client.upload(proxy)
    result = client.generate("analysis", WindowAnalysis, [video_part(file, 0, 6, 1.0), text_part(
        "Describe the visible events in this synthetic test pattern with source timestamps.")],
        system=ANALYSIS_SYSTEM, max_output_tokens=2048, resolution="low", thinking="low")
    assert all(0 <= e.start <= e.end <= 7 for e in result.events)
    report = client.ledger.report()
    assert report["calls"][-1]["prompt_tokens"] > 0
