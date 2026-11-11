"""Offline contract checks against the installed google-genai SDK (no network)."""
from types import SimpleNamespace

import pytest

genai = pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402

from shorts_pipeline.gemini import schemas  # noqa: E402
from shorts_pipeline.gemini.client import SdkTransport  # noqa: E402

ALL = [schemas.WindowAnalysis, schemas.VerificationResult, schemas.StoryMap, schemas.EditorialPlan,
       schemas.ScriptDraft, schemas.ScriptRevision, schemas.BeatRewrite, schemas.MatchingResult, schemas.QualityReview]


@pytest.mark.parametrize("schema", ALL, ids=lambda s: s.__name__)
def test_every_response_schema_converts_for_the_gemini_api(schema):
    from google.genai import _transformers
    client = genai.Client(api_key="test-key-not-used")
    converted = _transformers.t_schema(client._api_client, schema)
    assert converted.type == types.Type.OBJECT and converted.properties


def test_transport_builds_clipped_video_parts_and_structured_config(monkeypatch):
    transport = SdkTransport("test-key-not-used", timeout_seconds=30)
    captured = {}

    def fake_generate(model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        usage = SimpleNamespace(prompt_token_count=1000, cached_content_token_count=100, candidates_token_count=50,
                                thoughts_token_count=20, prompt_tokens_details=[
                                    SimpleNamespace(modality="MediaModality.AUDIO", token_count=200)])
        return SimpleNamespace(text='{"verdict":"pass","issues":[]}', usage_metadata=usage,
                               candidates=[SimpleNamespace(finish_reason="FinishReason.STOP")])
    monkeypatch.setattr(transport.client.models, "generate_content", fake_generate)
    part = {"file_uri": "https://generativelanguage.googleapis.com/v1beta/files/x", "mime_type": "video/mp4",
            "start": 12.5, "end": 30.0, "fps": 2.0}
    for model, field in (("gemini-2.5-flash", "thinking_budget"), ("gemini-3.8-flash", "thinking_level")):
        result = transport.generate(model, [part, {"text": "review"}], system="sys", schema=schemas.QualityReview,
                                    temperature=0.1, max_output_tokens=512, resolution="low", thinking="medium",
                                    cached_content=None)
        config = captured["config"]
        assert getattr(config.thinking_config, field) is not None
    video = captured["contents"][0].parts[0]
    assert video.file_data.mime_type == "video/mp4"
    assert video.video_metadata.start_offset == "12.5s" and video.video_metadata.end_offset == "30.0s"
    assert video.video_metadata.fps == 2.0
    assert config.response_mime_type == "application/json" and config.response_schema is schemas.QualityReview
    assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW
    assert config.system_instruction == "sys"
    assert result == {"text": '{"verdict":"pass","issues":[]}', "finish_reason": "STOP", "prompt_tokens": 1000,
                      "cached_tokens": 100, "output_tokens": 50, "thinking_tokens": 20, "audio_tokens": 200}


def test_sdk_errors_are_classified_for_retry_and_reporting():
    from google.genai import errors
    transport = SdkTransport("test-key-not-used", timeout_seconds=30)
    cases = {400: ("request", {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad"}}),
             401: ("auth", {"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "no"}}),
             404: ("model", {"error": {"code": 404, "status": "NOT_FOUND", "message": "models/x is not found"}}),
             429: ("rate_limit", {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}}),
             503: ("server", {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}})}
    for code, (category, body) in cases.items():
        error = (errors.ClientError if code < 500 else errors.ServerError)(code, body)
        assert transport.classify(error)[0] == category
    invalid_key = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                     "message": "API key not valid. Please pass a valid API key."}})
    assert transport.classify(invalid_key)[0] == "auth"
