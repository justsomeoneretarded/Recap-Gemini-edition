"""Presets, per-stage model preferences, and list-price estimates.

Model IDs are never assumed to exist: each stage lists ordered candidates and
`resolve_models` picks the first one the API key can actually see. Prices are
estimates for budgeting only; Google AI Studio billing is authoritative.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

STAGES = ("analysis", "planning", "writing", "matching", "review")

# Newest first. The SDK's own model catalogue (google-genai 3.1) lists these IDs;
# availability for a given key is checked at runtime.
FLASH = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
         "gemini-3-flash-preview", "gemini-2.5-flash")
PRO = ("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro")
LITE = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite")

PRESETS = {
    # Gemini API free tier: no billing, but low per-minute and per-day request
    # limits, and some (Pro) models may have no free quota. Flash models first,
    # Flash-Lite as fallbacks; calls are paced and capped per run.
    "free": {
        "models": {stage: FLASH + LITE for stage in ("analysis", "planning", "writing", "matching", "review")},
        "analysis_quality": "standard", "review_passes": 1, "max_budget_usd": 50.0,
        "render_review": False, "explicit_cache": False, "free_tier": True,
        "requests_per_minute": 8, "max_requests": 150, "zoom_cap": 6,
    },
    "quality": {
        "models": {"analysis": PRO, "planning": PRO, "writing": PRO, "matching": FLASH, "review": PRO},
        "analysis_quality": "detailed", "review_passes": 2, "max_budget_usd": 8.0,
        "render_review": True, "explicit_cache": True,
    },
    "balanced": {
        "models": {"analysis": FLASH, "planning": FLASH, "writing": PRO, "matching": FLASH, "review": FLASH},
        "analysis_quality": "standard", "review_passes": 2, "max_budget_usd": 3.0,
        "render_review": "on_issues", "explicit_cache": True,
    },
    "economy": {
        "models": {"analysis": LITE, "planning": LITE, "writing": FLASH, "matching": LITE, "review": FLASH},
        "analysis_quality": "fast", "review_passes": 1, "max_budget_usd": 1.0,
        "render_review": False, "explicit_cache": False,
    },
}

# fps/resolution of the coarse pass, window length, and conditional zoom checks.
ANALYSIS_QUALITY = {
    "fast": {"fps": 0.5, "resolution": "low", "window": 600.0, "zoom_checks": 0, "zoom_fps": 4.0,
             "zoom_resolution": "low", "match_fps": 1.0, "review_fps": 1.0},
    "standard": {"fps": 1.0, "resolution": "low", "window": 300.0, "zoom_checks": 10, "zoom_fps": 6.0,
                 "zoom_resolution": "medium", "match_fps": 2.0, "review_fps": 2.0},
    "detailed": {"fps": 1.0, "resolution": "medium", "window": 180.0, "zoom_checks": 30, "zoom_fps": 10.0,
                 "zoom_resolution": "high", "match_fps": 3.0, "review_fps": 2.0},
}

# Video token rates per sampled frame by media resolution, plus audio per second.
# Used only for pre-call budget reservations; actual usage comes from the API.
FRAME_TOKENS = {"low": 70, "medium": 260, "high": 260}
AUDIO_TOKENS_PER_SECOND = 32

# USD per 1M tokens: input, output (incl. thinking), cached input, audio input.
# Published paid-tier list prices for <=200k-token prompts as known to this
# repository. Entries marked assumed are conservative placeholders, not quotes.
PRICES = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cached": 0.125, "audio": 1.25, "source": "list price"},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.03, "audio": 1.00, "source": "list price"},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached": 0.01, "audio": 0.30, "source": "list price"},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.0, "cached": 0.20, "audio": 2.00, "source": "list price"},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached": 0.05, "audio": 1.00, "source": "list price"},
}
TIER_FALLBACK = {
    "lite": {"input": 0.30, "output": 2.00, "cached": 0.03, "audio": 1.00, "source": "assumed conservative lite-tier price"},
    "flash": {"input": 1.00, "output": 6.00, "cached": 0.10, "audio": 2.00, "source": "assumed conservative flash-tier price"},
    "pro": {"input": 4.00, "output": 20.0, "cached": 0.40, "audio": 4.00, "source": "assumed conservative pro-tier price"},
}


def price_for(model: str) -> dict:
    override = os.getenv("SHORTS_GEMINI_PRICES", "").strip()
    if override:
        try:
            table = json.loads(override)
            if model in table:
                return {**table[model], "source": "SHORTS_GEMINI_PRICES"}
        except ValueError as exc:
            raise ValueError("SHORTS_GEMINI_PRICES must be JSON: {model: {input, output, cached, audio}}") from exc
    if model in PRICES:
        return PRICES[model]
    if "lite" in model:
        return TIER_FALLBACK["lite"]
    if "flash" in model:
        return TIER_FALLBACK["flash"]
    return TIER_FALLBACK["pro"]


# Measured on the repository's 13-word validation sentence (docs/validation.md):
# original Chatterbox Warguy 4.72s, Kit Laser 4.80s, Female 4.44s; Nano 4.76/4.44/4.88s.
MEASURED_WORDS_PER_SECOND = {
    ("warguy", "original"): 13 / 4.72, ("kit-laser", "original"): 13 / 4.80, ("female", "original"): 13 / 4.44,
    ("warguy", "nano"): 13 / 4.76, ("kit-laser", "nano"): 13 / 4.44, ("female", "nano"): 13 / 4.88,
}
# Long passages contain pauses between sentences/beats that a 13-word clip lacks.
PLANNING_RATE_FACTOR = 0.86


def planning_rate(narrator: str, tts_model: str) -> float:
    measured = MEASURED_WORDS_PER_SECOND.get((narrator, tts_model))
    if measured is None:  # Turbo is unmeasured; use the slowest measured voice.
        measured = min(v for (n, _), v in MEASURED_WORDS_PER_SECOND.items() if n == narrator)
    return measured * PLANNING_RATE_FACTOR


@dataclass
class GeminiSettings:
    preset: str
    model_preferences: dict
    analysis_quality: str
    review_passes: int
    max_budget_usd: float
    render_review: object
    explicit_cache: bool
    strict_review: bool = False
    request_timeout_seconds: int = 600
    max_retries: int = 4
    max_output_repairs: int = 1
    max_rewrite_attempts: int = 2
    max_timing_repairs: int = 2
    max_source_minutes: float = 45.0
    free_tier: bool = False
    requests_per_minute: float | None = None
    max_requests: int | None = None
    zoom_cap: int | None = None
    resolved_models: dict = field(default_factory=dict)

    @property
    def analysis(self) -> dict:
        settings = dict(ANALYSIS_QUALITY[self.analysis_quality])
        if self.zoom_cap is not None:
            settings["zoom_checks"] = min(settings["zoom_checks"], self.zoom_cap)
        return settings

    def to_json(self) -> dict:
        return {"preset": self.preset, "analysis_quality": self.analysis_quality,
                "review_passes": self.review_passes, "max_budget_usd": self.max_budget_usd,
                "render_review": self.render_review, "explicit_cache": self.explicit_cache,
                "strict_review": self.strict_review, "free_tier": self.free_tier,
                "requests_per_minute": self.requests_per_minute, "max_requests": self.max_requests,
                "model_preferences": {k: list(v) for k, v in self.model_preferences.items()},
                "resolved_models": self.resolved_models, "analysis_settings": self.analysis}


def settings_from_request(request: dict) -> GeminiSettings:
    preset_name = request.get("gemini_preset", "free")
    if preset_name not in PRESETS:
        raise ValueError(f"gemini_preset must be one of {', '.join(PRESETS)}.")
    preset = PRESETS[preset_name]
    preferences = {stage: tuple(models) for stage, models in preset["models"].items()}
    overrides = request.get("gemini_models") or {}
    if not isinstance(overrides, dict) or any(k not in STAGES for k in overrides):
        raise ValueError(f"gemini_models may override only: {', '.join(STAGES)}.")
    for stage, value in overrides.items():
        models = (value,) if isinstance(value, str) else tuple(value)
        if not models or not all(isinstance(m, str) and m.strip() for m in models):
            raise ValueError(f"gemini_models.{stage} must name one or more model IDs.")
        preferences[stage] = tuple(m.strip() for m in models)
    quality = request.get("analysis_quality") or preset["analysis_quality"]
    if quality not in ANALYSIS_QUALITY:
        raise ValueError(f"analysis_quality must be one of {', '.join(ANALYSIS_QUALITY)}.")
    passes = request.get("review_passes", preset["review_passes"])
    if isinstance(passes, bool) or not isinstance(passes, int) or not 0 <= passes <= 4:
        raise ValueError("review_passes must be an integer from 0 to 4.")
    budget = float(request.get("max_budget_usd", preset["max_budget_usd"]))
    if not 0 < budget <= 50:
        raise ValueError("max_budget_usd must be greater than 0 and at most 50.")
    rpm = request.get("requests_per_minute", preset.get("requests_per_minute"))
    if rpm is not None and not 0 < float(rpm) <= 1000:
        raise ValueError("requests_per_minute must be between 0 and 1000.")
    cap = request.get("max_requests", preset.get("max_requests"))
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 5000):
        raise ValueError("max_requests must be an integer from 1 to 5000.")
    return GeminiSettings(preset=preset_name, model_preferences=preferences, analysis_quality=quality,
                          review_passes=passes, max_budget_usd=budget,
                          render_review=request.get("render_review", preset["render_review"]),
                          explicit_cache=bool(request.get("explicit_cache", preset["explicit_cache"])),
                          strict_review=bool(request.get("strict_review", False)),
                          max_source_minutes=float(request.get("max_source_minutes", 45.0)),
                          free_tier=bool(preset.get("free_tier", False)),
                          requests_per_minute=float(rpm) if rpm is not None else None,
                          max_requests=cap, zoom_cap=preset.get("zoom_cap"),
                          max_retries=6 if preset.get("free_tier") else 4)
