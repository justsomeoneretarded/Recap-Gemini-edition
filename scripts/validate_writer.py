"""Probe a real writer against a frozen production case without loading video models.

This deliberately replays saved observations. Passing means the production
normalizer accepts the result's structure and timing, not editorial quality or
an end-to-end source-to-video run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import requests

from shorts_pipeline import planner


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_model(base: str, model: str, output: Path) -> dict:
    deadline = time.monotonic() + 60
    while True:
        try:
            version = planner._request_json(base + "/api/version", timeout=3)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Ollama did not start; inspect ollama.log.")
            time.sleep(1)
    print(f"Pulling writer {model}", flush=True)
    last_state = None
    try:
        with requests.post(base + "/api/pull", json={"model": model, "stream": True},
                           stream=True, timeout=(10, 1800)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                last_state = json.loads(line)
                if last_state.get("error"):
                    raise RuntimeError(f"Model pull failed: {last_state['error']}")
        if not last_state or last_state.get("status") != "success":
            raise RuntimeError("Model pull ended without a success status.")
    finally:
        write_json(output / "model-pull.json", {"model": model, "last_state": last_state})
    tags = planner._request_json(base + "/api/tags", timeout=30)
    metadata = {"requested_model": model, "ollama": version,
                "installed_models": tags.get("models", []),
                "model_info": planner._request_json(base + "/api/show", {"model": model}, timeout=60)}
    write_json(output / "model-info.json", metadata)
    return metadata


def compress(candidate: dict, fixture: dict, request: dict, base: str,
             model: str, output: Path, label: str) -> dict:
    plan, attempts, error = planner._compress_narration(
        candidate, scenes=fixture["scenes"], described=fixture["described_scenes"],
        source=Path("snapshot-source-not-loaded.mp4"), request=request,
        duration=fixture["duration_seconds"], source_speech=[], base=base, writer=model,
        diagnostic_prefix=output / "diagnostics" / label, max_attempts=2)
    if plan is None:
        raise RuntimeError(f"Compression rejected after {attempts} attempt(s): {error}")
    return {"plan": plan, "repair_attempts": attempts}


def run_probe(fixture: dict, model: str, base: str, output: Path) -> dict:
    request = {**fixture["request"], "script_model": model}
    summary = {"state": "running", "scope": fixture["scope"], "model": model,
               "planner_version": planner.PLAN_VERSION, "fixture_provenance": fixture["provenance"],
               "full_writer": {"state": "pending"}, "saved_repair": {"state": "pending"}}
    write_json(output / "summary.json", summary)
    for label in ("full_writer", "saved_repair"):
        started = time.monotonic()
        summary[label] = {"state": "running"}
        write_json(output / "summary.json", summary)
        print(f"Writer probe: {label}", flush=True)
        evidence = {"model": model, "test": label, "raw_response": None, "parsed_candidate": None,
                    "output_schema": planner.PLAN_SCHEMA if label == "full_writer" else None,
                    "validation_error": None}
        try:
            if label == "full_writer":
                prompt = fixture["writer_prompt"]
                evidence["prompt"] = prompt
                estimated = math.ceil((len(prompt.encode("utf-8")) + len(json.dumps(planner.PLAN_SCHEMA).encode("utf-8"))) / 3)
                evidence["estimated_input_tokens"] = estimated
                if estimated > 12000:
                    raise ValueError(f"Saved writer prompt exceeds the 12000-token admission budget ({estimated}).")
                response = planner._request_json(base + "/api/generate", {
                    "model": model, "prompt": prompt, "format": planner.PLAN_SCHEMA,
                    "stream": False, "think": False, "keep_alive": "5m",
                    "options": {"temperature": .35, "num_predict": 4096, "num_ctx": 16384},
                }, timeout=1200)
                evidence["raw_response"] = response
                write_json(output / f"{label}.json", evidence)
                candidate = json.loads(response.get("response", ""))
                evidence["parsed_candidate"] = candidate
                if response.get("done_reason") in {"length", "max_tokens"}:
                    raise ValueError("Writer response was truncated; no plan was accepted.")
                try:
                    plan = planner._normalize_model_plan(candidate, fixture["scenes"],
                        Path("snapshot-source-not-loaded.mp4"), request, fixture["duration_seconds"], source_speech=[])
                    result = {"plan": plan, "repair_attempts": 0}
                except planner._NarrationTimingExcess as exc:
                    evidence["initial_validation_error"] = str(exc)
                    result = compress(candidate, fixture, request, base, model, output, "full-writer-repair")
            else:
                # Always invoke a real repair API call on the known feasible
                # overlong candidate, even if the fresh writer was accepted.
                evidence["original_candidate"] = fixture["saved_overlong_candidate"]
                result = compress(fixture["saved_overlong_candidate"], fixture, request,
                                  base, model, output, "saved-candidate-repair")
            result["plan"]["probe_scope"] = "Frozen observations only; source media was not loaded."
            write_json(output / f"{label}-accepted-plan.json", result["plan"])
            evidence.update(state="passed_admission", **result)
            summary[label] = {"state": "passed_admission", "repair_attempts": result["repair_attempts"],
                              "script": [beat["text"] for beat in result["plan"]["beats"]]}
        except Exception as exc:
            evidence.update(state="failed", validation_error=str(exc), error_type=type(exc).__name__)
            summary[label] = {"state": "failed", "error": str(exc), "error_type": type(exc).__name__}
        finally:
            evidence["elapsed_seconds"] = round(time.monotonic() - started, 3)
            summary[label]["elapsed_seconds"] = evidence["elapsed_seconds"]
            write_json(output / f"{label}.json", evidence)
            write_json(output / "summary.json", summary)
        print(f"Writer probe: {label} {summary[label]['state']}", flush=True)
    summary["state"] = ("succeeded" if all(summary[label]["state"] == "passed_admission"
                                         for label in ("full_writer", "saved_repair")) else "failed")
    summary["limitations"] = ["No fresh footage/vision, speech synthesis or rendering was run.",
                              "Passing structure and timing checks does not establish narrative quality or source accuracy."]
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("WRITER_MODEL", "qwen3.5:4b-q4_K_M"))
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "docs/validation-fixtures/writer-case.json")
    parser.add_argument("--output", type=Path, default=Path("output/writer-probe"))
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,199}", args.model):
        parser.error("Provide a non-empty Ollama model identifier.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Choose an empty output directory to preserve previous probe results.")
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    base = args.base_url.rstrip("/")
    write_json(output / "fixture-identity.json", {"sha256": hashlib.sha256(fixture_bytes).hexdigest(),
                                                "fixture_version": fixture["fixture_version"]})
    try:
        prepare_model(base, args.model, output)
        result = run_probe(fixture, args.model, base, output)
    except Exception as exc:
        result = {"state": "failed", "stage": "setup", "error": str(exc),
                  "error_type": type(exc).__name__, "model": args.model, "scope": fixture["scope"]}
        write_json(output / "summary.json", result)
    print(json.dumps({"state": result["state"], "model": args.model, "output": str(output)}), flush=True)
    return 0 if result["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
