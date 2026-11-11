"""Multi-stage, evidence-grounded Gemini production pipeline.

Stages (see docs/gemini-architecture.md):
A analysis   - native video understanding of source windows + targeted zoom checks
B story map  - evidence-backed event graph (observations vs. inferences)
C editorial  - which events make a coherent, engaging recap
D script     - narration beats with evidence references, then a review pass
E matching   - sentence-to-shot evidence map using visual reasoning
F shot plan  - deterministic in/out, continuity, reuse and pacing rules
G sync       - measured narration, sentence-level frame allocation
H review     - independent deterministic + Gemini quality review
I correction - bounded, targeted repairs with re-checks
J verify     - checks on the actual rendered MP4
"""
