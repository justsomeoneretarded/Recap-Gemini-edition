"""Write a short, readable run summary to the GitHub Actions job page."""
import json
import os
from pathlib import Path

out = Path('output/shorts')


def load(name):
    try:
        return json.loads((out/name).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


status, cost, quality, verify = load('status.json'), load('cost-report.json'), load('quality-review.json'), load('final-verification.json')
lines = ['## Generate Shorts result', '',
         f"- **State:** {status.get('state', 'unknown')}" + (f" — {status.get('error','')[:300]}" if status.get('error') else ''),
         f"- **Duration:** {status.get('duration', '–')} s",
         f"- **Pre-render review:** {quality.get('review_status', status.get('review_status', 'n/a'))} "
         f"({quality.get('counts', {})})",
         f"- **Rendered-file checks:** {verify.get('automated_verdict', 'n/a')} — human playback review is still required",
         (f"- **Gemini usage:** free tier, not billed — {cost.get('requests_sent', 0)} requests "
          f"(paid list-price equivalent ${cost.get('estimated_total_usd', 0):.4f})"
          if cost.get('max_budget_usd') is None and cost else
          f"- **Estimated Gemini cost:** ${cost.get('estimated_total_usd', 0):.4f} of ${cost.get('max_budget_usd') or 0:.2f} budget "
          "(estimate from token usage; AI Studio billing is authoritative)")]
for warning in quality.get('source_warnings', []):
    lines.append(f"- **Source warning:** {warning}")
if status.get('fallback'):
    lines.append(f"- **Fallback used:** {status['fallback']['writer']} because {status['fallback']['reason'][:200]}")
if cost.get('by_stage'):
    lines += ['', '| Stage | Models | Calls (cached) | Input tokens | Output tokens | Est. USD | Retries |', '|---|---|---|---|---|---|---|']
    for stage, row in cost['by_stage'].items():
        lines.append(f"| {stage} | {', '.join(row['models'])} | {row['calls']} ({row['cache_hits']}) | {row['prompt_tokens']} "
                     f"| {row['output_tokens'] + row['thinking_tokens']} | {row['estimated_cost_usd']:.4f} | {row['retries']} |")
critical = [i for i in quality.get('issues', []) if i.get('severity') == 'critical']
if critical:
    lines += ['', '### Unresolved critical issues'] + [f"- {i['check']}: {i['description'][:200]}" for i in critical[:10]]
summary = os.environ.get('GITHUB_STEP_SUMMARY')
text = '\n'.join(lines) + '\n'
if summary:
    with open(summary, 'a', encoding='utf-8') as stream:
        stream.write(text)
print(text)
