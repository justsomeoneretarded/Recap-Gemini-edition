"""Gemini access through the official google-genai SDK.

Responsibilities: key handling (never logged), model availability checks,
structured JSON output validated by Pydantic, bounded retries with recorded
reasons, per-run budget enforcement, response/upload/context caching, and a
token/cost ledger. A `transport` object isolates the SDK so ordinary tests run
fully offline with scripted fakes.
"""
from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from .settings import AUDIO_TOKENS_PER_SECOND, FRAME_TOKENS, GeminiSettings, price_for

CLIENT_VERSION = 1


class GeminiError(RuntimeError):
    """Base class; messages never include the API key."""


class GeminiAuthError(GeminiError):
    pass


class GeminiModelUnavailable(GeminiError):
    pass


class GeminiRateLimited(GeminiError):
    pass


class GeminiTimeout(GeminiError):
    pass


class GeminiOutputError(GeminiError):
    pass


class BudgetExceeded(GeminiError):
    pass


def _redact(text: str) -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    text = str(text)
    if key and len(key) >= 8:
        text = text.replace(key, "[REDACTED]")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


# ------------------------------------------------------------------ parts
def text_part(text: str) -> dict:
    return {"text": text}


def video_part(file: "UploadedFile", start: float | None = None, end: float | None = None,
               fps: float | None = None) -> dict:
    part = {"file_uri": file.uri, "mime_type": file.mime_type, "file_sha256": file.sha256}
    if start is not None:
        part["start"] = round(max(0.0, float(start)), 3)
    if end is not None:
        part["end"] = round(float(end), 3)
    if fps is not None:
        part["fps"] = float(fps)
    return part


def estimate_tokens(parts: list[dict], resolution: str = "low", system: str = "") -> int:
    tokens = math.ceil(len(system) / 3.5)
    for part in parts:
        if "text" in part:
            tokens += math.ceil(len(part["text"]) / 3.5)
        else:
            seconds = max(0.0, float(part.get("end", part.get("duration", 60))) - float(part.get("start", 0)))
            fps = float(part.get("fps", 1.0))
            tokens += math.ceil(seconds * (fps * FRAME_TOKENS.get(resolution, 260) + AUDIO_TOKENS_PER_SECOND))
    return tokens


# ------------------------------------------------------------------ ledger
class CostLedger:
    """Token usage, estimated cost, retries and stage timing for one run."""

    def __init__(self, max_budget_usd: float, path: Path | None = None):
        self.max_budget_usd = float(max_budget_usd)
        self.path = path
        self.calls: list[dict] = []
        self.stage_seconds: dict[str, float] = {}
        self.events: list[dict] = []

    @property
    def spent_usd(self) -> float:
        return sum(c["estimated_cost_usd"] for c in self.calls)

    def cost(self, model: str, prompt: int, output: int, cached: int = 0, audio: int = 0) -> float:
        price = price_for(model)
        uncached = max(0, prompt - cached)
        audio = min(audio, uncached)
        return ((uncached - audio) * price["input"] + audio * price.get("audio", price["input"])
                + cached * price["cached"] + output * price["output"]) / 1e6

    def reserve(self, stage: str, model: str, estimated_input: int, max_output: int) -> float:
        estimate = self.cost(model, estimated_input, max_output)
        if self.spent_usd + estimate > self.max_budget_usd + 1e-9:
            raise BudgetExceeded(
                f"Stage {stage} would exceed the ${self.max_budget_usd:.2f} Gemini budget: "
                f"${self.spent_usd:.4f} already used plus up to ${estimate:.4f} estimated for {model}. "
                "Raise max_budget_usd, choose the economy preset, or use a shorter source window.")
        return estimate

    def record(self, **entry) -> None:
        self.calls.append(entry)
        self.save()

    def note(self, **event) -> None:
        self.events.append(event)
        self.save()

    def time_stage(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] = round(self.stage_seconds.get(stage, 0.0) + seconds, 3)
        self.save()

    def report(self) -> dict:
        by_stage: dict[str, dict] = {}
        for call in self.calls:
            row = by_stage.setdefault(call["stage"], {"calls": 0, "cache_hits": 0, "models": set(),
                                                      "prompt_tokens": 0, "cached_tokens": 0,
                                                      "output_tokens": 0, "thinking_tokens": 0,
                                                      "estimated_cost_usd": 0.0, "retries": 0,
                                                      "retry_reasons": []})
            row["calls"] += 1
            row["cache_hits"] += call.get("response_cache_hit", False)
            row["models"].add(call["model"])
            for key in ("prompt_tokens", "cached_tokens", "output_tokens", "thinking_tokens", "estimated_cost_usd"):
                row[key] += call.get(key, 0) or 0
            row["retries"] += len(call.get("retry_reasons", []))
            row["retry_reasons"] += call.get("retry_reasons", [])
        for row in by_stage.values():
            row["models"] = sorted(row["models"])
            row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 6)
        return {
            "note": ("Costs are estimates from token usage reported by the API and the price table in "
                     "shorts_pipeline/gemini/settings.py (or SHORTS_GEMINI_PRICES). Google AI Studio "
                     "billing is authoritative; the API does not return billed amounts."),
            "max_budget_usd": self.max_budget_usd,
            "estimated_total_usd": round(self.spent_usd, 6),
            "by_stage": by_stage,
            "stage_seconds": self.stage_seconds,
            "price_sources": {m: price_for(m).get("source") for m in sorted({c["model"] for c in self.calls})},
            "calls": self.calls,
            "events": self.events,
        }

    def save(self) -> None:
        if self.path:
            _write_json(self.path, self.report())


# ------------------------------------------------------------------ transport
@dataclass
class UploadedFile:
    name: str
    uri: str
    mime_type: str
    sha256: str


class SdkTransport:
    """Thin adapter over google-genai. Imported lazily so offline paths need no SDK."""

    def __init__(self, api_key: str, timeout_seconds: int):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiError("The Gemini pipeline needs the google-genai package: "
                              "pip install -e '.[gemini]'.") from exc
        self.types = types
        self.errors = __import__("google.genai.errors", fromlist=["errors"])
        # Retries are handled by GeminiClient so every reason is recorded.
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout_seconds * 1000), retry_options=types.HttpRetryOptions(attempts=1)))

    def classify(self, exc: Exception) -> tuple[str, str]:
        """Return (category, safe message) for an SDK/network exception."""
        message = _redact(str(exc))
        if isinstance(exc, self.errors.APIError):
            code, status = getattr(exc, "code", None), str(getattr(exc, "status", "") or "")
            lowered = message.lower()
            if code in (401, 403) or "api key" in lowered or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                return "auth", message
            if code == 404 or status == "NOT_FOUND":
                return "model", message
            if code == 429 or status == "RESOURCE_EXHAUSTED":
                return "rate_limit", message
            if code in (408, 504) or status == "DEADLINE_EXCEEDED":
                return "timeout", message
            if code and code >= 500:
                return "server", message
            return "request", message
        name = type(exc).__name__.lower()
        if "timeout" in name:
            return "timeout", message
        if "connect" in name or "network" in name or "remoteprotocol" in name:
            return "server", message
        return "request", message

    def list_models(self) -> list[str]:
        names = []
        for model in self.client.models.list():
            names.append(str(model.name).removeprefix("models/"))
        return names

    def _part(self, part: dict):
        t = self.types
        if "text" in part:
            return t.Part(text=part["text"])
        metadata = None
        if any(k in part for k in ("start", "end", "fps")):
            metadata = t.VideoMetadata(
                start_offset=f"{part['start']}s" if "start" in part else None,
                end_offset=f"{part['end']}s" if "end" in part else None,
                fps=part.get("fps"))
        return t.Part(file_data=t.FileData(file_uri=part["file_uri"], mime_type=part["mime_type"]),
                      video_metadata=metadata)

    def _thinking(self, model: str, level: str | None):
        if not level:
            return None
        if model.startswith("gemini-2."):
            return self.types.ThinkingConfig(thinking_budget={"low": 1024, "medium": 4096, "high": 12288}[level])
        return self.types.ThinkingConfig(thinking_level=level.upper())

    def generate(self, model: str, parts: list[dict], *, system: str, schema: type[BaseModel],
                 temperature: float, max_output_tokens: int, resolution: str | None,
                 thinking: str | None, cached_content: str | None):
        t = self.types
        config = t.GenerateContentConfig(
            system_instruction=None if cached_content else system,
            response_mime_type="application/json", response_schema=schema,
            temperature=temperature, max_output_tokens=max_output_tokens,
            media_resolution={"low": "MEDIA_RESOLUTION_LOW", "medium": "MEDIA_RESOLUTION_MEDIUM",
                              "high": "MEDIA_RESOLUTION_HIGH"}.get(resolution or ""),
            thinking_config=self._thinking(model, thinking), cached_content=cached_content)
        response = self.client.models.generate_content(
            model=model, contents=[t.Content(role="user", parts=[self._part(p) for p in parts])], config=config)
        usage = response.usage_metadata
        audio = 0
        for detail in (getattr(usage, "prompt_tokens_details", None) or []):
            if str(getattr(detail, "modality", "")).upper().endswith("AUDIO"):
                audio += detail.token_count or 0
        finish = None
        if response.candidates:
            finish = str(response.candidates[0].finish_reason or "").rsplit(".", 1)[-1]
        return {"text": response.text or "", "finish_reason": finish,
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "cached_tokens": getattr(usage, "cached_content_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
                "audio_tokens": audio}

    def upload(self, path: Path, mime_type: str) -> dict:
        file = self.client.files.upload(file=str(path), config=self.types.UploadFileConfig(
            mime_type=mime_type, display_name=Path(path).name))
        return self._file(file)

    def get_file(self, name: str) -> dict:
        return self._file(self.client.files.get(name=name))

    def _file(self, file) -> dict:
        state = str(getattr(file, "state", "") or "").rsplit(".", 1)[-1]
        expires = getattr(file, "expiration_time", None)
        return {"name": file.name, "uri": file.uri, "mime_type": file.mime_type, "state": state,
                "expires_epoch": expires.timestamp() if expires else None}

    def create_cache(self, model: str, parts: list[dict], system: str, ttl_seconds: int) -> str:
        t = self.types
        cache = self.client.caches.create(model=model, config=t.CreateCachedContentConfig(
            contents=[t.Content(role="user", parts=[self._part(p) for p in parts])],
            system_instruction=system, ttl=f"{int(ttl_seconds)}s", display_name="recap-shorts-evidence"))
        return cache.name


# ------------------------------------------------------------------ client
class GeminiClient:
    def __init__(self, settings: GeminiSettings, ledger: CostLedger, cache_dir: Path,
                 transport=None, sleep: Callable[[float], None] | None = None):
        self.settings, self.ledger = settings, ledger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep or (lambda seconds: time.sleep(seconds))
        if transport is None:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise GeminiAuthError(
                    "GEMINI_API_KEY is not set. Create a key in Google AI Studio and add it as the "
                    "GEMINI_API_KEY repository secret (Actions) or a local environment variable. "
                    "Or run with planner=local to use the local Gemma pipeline.")
            transport = SdkTransport(key, settings.request_timeout_seconds)
        self.transport = transport
        self._context_caches: dict[str, str] = {}

    # ---- models
    def resolve_models(self) -> dict[str, str]:
        try:
            available = set(self.transport.list_models())
        except Exception as exc:  # noqa: BLE001 - classified below
            category, message = self._classify(exc)
            if category == "auth":
                raise GeminiAuthError(f"Gemini rejected the API key while listing models: {message}") from exc
            raise GeminiError(f"Could not list Gemini models ({category}): {message}") from exc
        resolved = {}
        for stage, candidates in self.settings.model_preferences.items():
            chosen = next((m for m in candidates if m in available), None)
            if chosen is None:
                raise GeminiModelUnavailable(
                    f"None of the configured {stage} models are available to this API key: "
                    f"{', '.join(candidates)}. Choose another preset or set gemini_models.{stage}.")
            if chosen != candidates[0]:
                self.ledger.note(kind="model_substitution", stage=stage, preferred=candidates[0], used=chosen)
            resolved[stage] = chosen
        self.settings.resolved_models = resolved
        return resolved

    def _classify(self, exc: Exception) -> tuple[str, str]:
        if hasattr(self.transport, "classify"):
            return self.transport.classify(exc)
        return "request", _redact(str(exc))

    # ---- files
    def upload(self, path: Path, mime_type: str | None = None) -> UploadedFile:
        """Upload once per file hash; reuse until shortly before the 48h expiry."""
        path = Path(path)
        digest = _sha256(path)
        mime_type = mime_type or mimetypes.guess_type(path.name)[0] or "video/mp4"
        registry_path = self.cache_dir / "uploads.json"
        registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
        previous = registry.get(digest)
        if previous and (previous.get("expires_epoch") or 0) > time.time() + 3600:
            try:
                info = self.transport.get_file(previous["name"])
                if info.get("state") == "ACTIVE":
                    self.ledger.note(kind="upload_reused", file=path.name, name=previous["name"])
                    return UploadedFile(previous["name"], previous["uri"], previous["mime_type"], digest)
            except Exception as exc:  # noqa: BLE001 - expired/deleted: upload again
                self.ledger.note(kind="upload_reuse_failed", reason=self._classify(exc)[1][:200])
        started = time.time()
        try:
            info = self.transport.upload(path, mime_type)
        except Exception as exc:  # noqa: BLE001
            category, message = self._classify(exc)
            error = GeminiAuthError if category == "auth" else GeminiError
            raise error(f"Uploading {path.name} to the Gemini Files API failed ({category}): {message}") from exc
        deadline = time.time() + 900
        while info.get("state") == "PROCESSING":
            if time.time() > deadline:
                raise GeminiTimeout(f"Gemini kept {path.name} in PROCESSING for 15 minutes.")
            self.sleep(5)
            info = self.transport.get_file(info["name"])
        if info.get("state") not in ("ACTIVE", "", None):
            raise GeminiError(f"Gemini could not process {path.name}: state {info.get('state')}.")
        registry[digest] = {"name": info["name"], "uri": info["uri"], "mime_type": info.get("mime_type") or mime_type,
                            "expires_epoch": info.get("expires_epoch") or time.time() + 47 * 3600,
                            "uploaded_epoch": time.time()}
        _write_json(registry_path, registry)
        self.ledger.note(kind="upload", file=path.name, bytes=path.stat().st_size,
                         seconds=round(time.time() - started, 2))
        return UploadedFile(info["name"], info["uri"], registry[digest]["mime_type"], digest)

    # ---- explicit context cache
    def context_cache(self, model: str, parts: list[dict], system: str, min_tokens: int = 4096) -> str | None:
        """Create an explicit cache for a large shared prefix; None if not worthwhile."""
        if not self.settings.explicit_cache or not hasattr(self.transport, "create_cache"):
            return None
        tokens = estimate_tokens(parts, system=system)
        if tokens < min_tokens:
            return None
        key = hashlib.sha256(json.dumps([model, system, self._stable(parts)], sort_keys=True).encode()).hexdigest()
        if key in self._context_caches:
            return self._context_caches[key]
        try:
            name = self.transport.create_cache(model, parts, system, ttl_seconds=3600)
        except Exception as exc:  # noqa: BLE001 - caching is an optimization only
            self.ledger.note(kind="explicit_cache_unavailable", model=model, reason=self._classify(exc)[1][:300])
            return None
        self.ledger.note(kind="explicit_cache_created", model=model, estimated_tokens=tokens)
        self._context_caches[key] = name
        return name

    @staticmethod
    def _stable(parts: list[dict]) -> list[dict]:
        # Identify uploaded media by content hash, not by expiring file URIs.
        return [{k: v for k, v in p.items() if k not in ("file_uri",)} for p in parts]

    # ---- generation
    def generate(self, stage: str, schema: type[BaseModel], parts: list[dict], *, system: str,
                 model: str | None = None, temperature: float = 0.3, max_output_tokens: int = 8192,
                 resolution: str | None = None, thinking: str | None = "medium",
                 cached_prefix: list[dict] | None = None) -> BaseModel:
        """Call Gemini for a validated structured result, with caching and retries.

        `cached_prefix` parts precede `parts`; when large they become an explicit
        context cache so repeated stages pay the cached-token rate.
        """
        model = model or self.settings.resolved_models.get(stage.split(":")[0]) or self.settings.model_preferences[stage.split(":")[0]][0]
        prefix = cached_prefix or []
        identity = hashlib.sha256(json.dumps({
            "v": CLIENT_VERSION, "model": model, "system": system, "schema": schema.__name__,
            "schema_json": schema.model_json_schema(), "parts": self._stable(prefix + parts),
            "temperature": temperature, "resolution": resolution, "thinking": thinking,
            "max_output_tokens": max_output_tokens}, sort_keys=True, default=str).encode()).hexdigest()
        cached_file = self.cache_dir / "responses" / f"{identity}.json"
        if cached_file.is_file():
            saved = json.loads(cached_file.read_text(encoding="utf-8"))
            try:
                value = schema.model_validate(saved["parsed"])
                self.ledger.record(stage=stage, model=model, response_cache_hit=True, prompt_tokens=0,
                                   cached_tokens=0, output_tokens=0, thinking_tokens=0,
                                   estimated_cost_usd=0.0, seconds=0.0, retry_reasons=[])
                return value
            except (KeyError, ValidationError):
                pass
        cache_name = self.context_cache(model, prefix, system) if prefix else None
        call_parts = parts if cache_name else prefix + parts
        estimate = estimate_tokens(call_parts if not cache_name else parts, resolution or "low", system)
        reasons: list[str] = []
        feedback = ""
        started = time.time()
        attempt = 0
        repairs = 0
        while True:
            attempt += 1
            self.ledger.reserve(stage, model, estimate, max_output_tokens)
            request_parts = call_parts + ([text_part(feedback)] if feedback else [])
            try:
                result = self.transport.generate(model, request_parts, system=system, schema=schema,
                                                 temperature=temperature, max_output_tokens=max_output_tokens,
                                                 resolution=resolution, thinking=thinking,
                                                 cached_content=cache_name)
            except Exception as exc:  # noqa: BLE001 - classified
                category, message = self._classify(exc)
                if category == "auth":
                    raise GeminiAuthError(f"Gemini rejected the API key during {stage}: {message}") from exc
                if category == "model":
                    raise GeminiModelUnavailable(f"Model {model} is not available for {stage}: {message}") from exc
                if category in ("rate_limit", "timeout", "server") and attempt <= self.settings.max_retries:
                    reasons.append(f"{category}: {message[:160]}")
                    delay = min(60.0, 2.0 * 2 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
                    self.sleep(delay)
                    continue
                failure = {"rate_limit": GeminiRateLimited, "timeout": GeminiTimeout}.get(category, GeminiError)
                self._record(stage, model, None, started, reasons, error=f"{category}: {message[:300]}")
                raise failure(f"Gemini {stage} failed after {attempt} attempt(s) ({category}): {message[:500]}") from exc
            self._record(stage, model, result, started, reasons)
            started = time.time()
            text = result.get("text") or ""
            try:
                if result.get("finish_reason") in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
                    raise GeminiOutputError(f"Gemini blocked the {stage} response ({result['finish_reason']}).")
                if result.get("finish_reason") == "MAX_TOKENS":
                    raise ValueError("response was truncated at the output token limit")
                parsed = schema.model_validate(json.loads(text))
            except GeminiOutputError:
                raise
            except (ValueError, ValidationError) as exc:
                if repairs >= self.settings.max_output_repairs:
                    diagnostic = self.cache_dir / "diagnostics" / f"{stage.replace(':', '-')}-{identity[:12]}.json"
                    _write_json(diagnostic, {"stage": stage, "model": model, "raw": text[:20000], "error": str(exc)[:2000]})
                    raise GeminiOutputError(f"Gemini returned malformed {schema.__name__} for {stage} "
                                            f"after {repairs + 1} attempts: {str(exc)[:400]}. Raw output: {diagnostic}") from exc
                repairs += 1
                reasons.append(f"malformed_output: {str(exc)[:160]}")
                feedback = ("Your previous response was not valid for the required JSON schema: "
                            f"{str(exc)[:800]}. Return only corrected JSON that matches the schema.")
                continue
            _write_json(cached_file, {"stage": stage, "model": model, "parsed": parsed.model_dump(mode="json")})
            return parsed

    def _record(self, stage, model, result, started, reasons, error=None):
        result = result or {}
        cost = self.ledger.cost(model, result.get("prompt_tokens", 0),
                                result.get("output_tokens", 0) + result.get("thinking_tokens", 0),
                                result.get("cached_tokens", 0), result.get("audio_tokens", 0))
        self.ledger.record(stage=stage, model=model, response_cache_hit=False,
                           prompt_tokens=result.get("prompt_tokens", 0), cached_tokens=result.get("cached_tokens", 0),
                           output_tokens=result.get("output_tokens", 0), thinking_tokens=result.get("thinking_tokens", 0),
                           audio_tokens=result.get("audio_tokens", 0), estimated_cost_usd=round(cost, 6),
                           seconds=round(time.time() - started, 3), retry_reasons=list(reasons),
                           finish_reason=result.get("finish_reason"), error=error)
        reasons.clear()
