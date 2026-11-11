# Graph Report - Recap-Short-form  (2026-11-06)

## Corpus Check
- 43 files · ~58,362 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 506 nodes · 1133 edges · 23 communities (13 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d0c29466`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_sources_planner.py
- default / description
- difflib / re
- array / copy
- generate-shorts.yml / concurrency
- argparse / pathlib
- Evidence-linked story beats and timed edit decisions / Source-grounded
- analyze_source_speech
- ci.yml / jobs
- sources.py
- recap-shorts
- write_captions
- with
- json

## God Nodes (most connected - your core abstractions)
1. `make_plan()` - 32 edges
2. `render_plan()` - 30 edges
3. `generate()` - 23 edges
4. `write_captions()` - 22 edges
5. `analyze_source_speech()` - 21 edges
6. `Eight reference Shorts: evidence and production guide` - 21 edges
7. `_compress_narration()` - 20 edges
8. `resolve_source()` - 18 edges
9. `synthesize_plan()` - 18 edges
10. `api()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `_normalize_speech()` --calls--> `_normalize_model_plan()`  [EXTRACTED]
  tests/test_sources_planner.py → shorts_pipeline/planner.py
- `test_bad_segment_limits_rejected()` --calls--> `_segments()`  [EXTRACTED]
  tests/test_sources_planner.py → shorts_pipeline/sources.py
- `test_compression_preserves_selected_native_quote_and_uses_remaining_narration_budget()` --calls--> `make_plan()`  [EXTRACTED]
  tests/test_sources_planner.py → shorts_pipeline/planner.py
- `test_compression_undershoot_exhausts_text_attempts_before_full_storyboard_retry()` --calls--> `make_plan()`  [EXTRACTED]
  tests/test_sources_planner.py → shorts_pipeline/planner.py
- `test_failed_download_is_not_cached()` --calls--> `resolve_source()`  [EXTRACTED]
  tests/test_sources_planner.py → shorts_pipeline/sources.py

## Import Cycles
- None detected.

## Communities (23 total, 10 thin omitted)

### Community 0 - "test_sources_planner.py"
Cohesion: 0.07
Nodes (50): validate_storyboard(), _is_compression(), _normalize_speech(), _plan(), _production_six_candidate(), _production_three_candidates(), _speech_candidate(), _speech_plan() (+42 more)

### Community 1 - "default / description"
Cohesion: 0.04
Nodes (45): default, description, options, type, description, required, type, aspect (+37 more)

### Community 2 - "difflib / re"
Cohesion: 0.10
Nodes (24): _align_script(), _chunks(), _contextual_units(), _integer_words(), _lexical(), _normal(), _run(), synthesize_plan() (+16 more)

### Community 3 - "array / copy"
Cohesion: 0.15
Nodes (26): allocate_shots(), _ass_escape(), _attach_narration(), _concat(), _covers(), crop_filter(), _ffmpeg(), _master_audio() (+18 more)

### Community 4 - "generate-shorts.yml / concurrency"
Cohesion: 0.05
Nodes (39): concurrency, cancel-in-progress, group, HF_HOME, OLLAMA_MAX_LOADED_MODELS, OLLAMA_MODELS, OLLAMA_NUM_PARALLEL, SCRIPT_MODEL (+31 more)

### Community 5 - "argparse / pathlib"
Cohesion: 0.12
Nodes (15): generate(), stage(), main(), assign_timeline(), request_config(), excerpt(), probe(), run() (+7 more)

### Community 6 - "Evidence-linked story beats and timed edit decisions / Source-grounded"
Cohesion: 0.07
Nodes (27): Eight reference Shorts: evidence and production guide, V1 — Everyone Wants Him Dead, V2 — He's Not a Cop at All, V3 — He Was Hired to Kill. Now He's the Prey, V4 — He Was the Target All Along, V5 — One Man vs the Deadliest Fighter, V6 — Rookie Cop Just Learned the Hard Truth, V7 — The Next Assassin Is Already on His Way—but This Father Is Ready (+19 more)

### Community 7 - "analyze_source_speech"
Cohesion: 0.12
Nodes (17): analyze_source_speech(), _candidate(), _number(), _transcribe(), fixture(), segment(), test_bad_word_bounds_and_confidence(), test_cache_and_content_invalidation() (+9 more)

### Community 8 - "ci.yml / jobs"
Cohesion: 0.12
Nodes (15): jobs, test, name, on, permissions, contents, name, run (+7 more)

### Community 9 - "sources.py"
Cohesion: 0.11
Nodes (13): _adaptive_boundaries(), _ffmpeg_boundaries(), file_digest(), _https_url(), index_source(), resolve_source(), _run(), _segments() (+5 more)

### Community 11 - "write_captions"
Cohesion: 0.19
Nodes (17): _ass_time(), _escape_ass(), _groups(), _srt_time(), _word_fade_text(), _wrap(), write_captions(), _events() (+9 more)

### Community 12 - "with"
Cohesion: 0.06
Nodes (33): OLLAMA_MAX_LOADED_MODELS, OLLAMA_MODELS, OLLAMA_NUM_PARALLEL, SCRIPT_MODEL, WRITER_MODEL, writer_model, jobs, probe (+25 more)

### Community 17 - "json"
Cohesion: 0.11
Nodes (23): compress(), main(), prepare_model(), run_probe(), write_json(), _compress_narration(), _compression_schema(), _describe() (+15 more)

## Knowledge Gaps
- **106 isolated node(s):** `name`, `on`, `contents`, `runs-on`, `timeout-minutes` (+101 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 192 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `make_plan()` connect `json` to `test_sources_planner.py`, `sources.py`, `array / copy`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **What connects `name`, `on`, `contents` to the rest of the system?**
  _106 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_sources_planner.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06573426573426573 - nodes in this community are weakly interconnected._
- **Why does `generate()` connect `argparse / pathlib` to `test_sources_planner.py`, `difflib / re`, `array / copy`, `analyze_source_speech`, `sources.py`, `write_captions`, `json`?**
  _High betweenness centrality (0.002) - this node is a cross-community bridge._
- **Should `default / description` be split into smaller, more focused modules?**
  _Cohesion score 0.044444444444444446 - nodes in this community are weakly interconnected._
- **Why does `render_plan()` connect `array / copy` to `json`, `sources.py`?**
  _High betweenness centrality (0.001) - this node is a cross-community bridge._
- **Should `difflib / re` be split into smaller, more focused modules?**
  _Cohesion score 0.0951219512195122 - nodes in this community are weakly interconnected._