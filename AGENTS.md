## graphify

Graphify is installed for this project. The project graph belongs in graphify-out/ and connects code with documented reference observations. Reference source files and measurements are in docs/reference-data/.

Workspace executable: `/workspace/work/graphify/bin/graphify` (version 0.9.102, with video dependencies). Use that absolute path, or prepend `/workspace/work/graphify/bin` to PATH for a command. The matching Python interpreter is `/workspace/work/graphify/tools/graphifyy/bin/python`. The isolated installation and caches live under `/workspace/work/graphify`, outside this repository. Set `GRAPHIFY_QUERY_LOG_DISABLE=1` when running queries to avoid writing logs to the home directory.

The agreed video workflow starts with direct examination of reference footage, audio measurements, sampled frames, and timestamped transcripts. Use Graphify to connect and analyze the resulting evidence across videos. Distinguish observations from inferred patterns and retain source timestamps. Generate Shorts is an independent workflow. See docs/validation.md for exact runtime coverage; do not imply complete frame-by-frame analysis, auditioned voice likeness, or a tested full default-model run unless newly verified.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
