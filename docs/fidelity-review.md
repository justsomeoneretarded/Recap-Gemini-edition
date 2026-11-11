Implementation update: automatic source-speech selection and progressive caption fades are now implemented and tested. The observations below document the earlier gaps and their evidence. Automatic object/face framing across a complete shot remains a separate quality priority. Full model-generated output still needs production validation.

# Standalone Shorts fidelity review

Reviewed 2026-11-05. Read-only code snapshot; the planner is being edited concurrently, so function names are more durable than line numbers. Scope is this Shorts pipeline and the eight supplied references. Top10 production behavior is not a requirement. Graphify was queried before code inspection. No project files or packages were changed.

The largest fidelity gap is editorial: the automatic route cannot yet make the reference pattern of narration interrupted by meaningful original dialogue. Visual grounding and caption behavior are the next concrete differences. The current square canvas, compact white italic captions, action-first script instructions, hard cuts, measured narration duration and varied ending allowance are useful foundations.

## 1. Automatic stories exclude original-dialogue passages

**Priority: high; code limitation certain, reference speaker attribution still needs audition.**

Reference evidence: V1 has apparent source speech at **22.36–23.78**, **68.20–72.70**, **82.86–92.34** and **121.38–126.56**. The long library exchange supplies the adversary's motive and deadline tension; narration resumes at **93.26–97.86** to compress the defeat. V5 begins with apparent dialogue at **0.00–6.98**, with narration entering **7.28–17.58**. V6 includes an apparent conversation around **83–104 seconds**. These observations come from timestamped ASR plus sampled imagery, not certified diarization. See `docs/reference-analysis.md:44`, `:51`, `:69`, `:79` and corresponding V1/V5/V6 transcripts.

Exact behavior: `planner.py::make_plan` explicitly says **“Use narration beats only”** (current line 294); `_normalize_model_plan` unconditionally emits `kind: narration` (217–219). Its coverage metadata correctly states no dialogue analysis (231). `render.py` can preserve a manually supplied dialogue beat at original speed (259–285), so the missing component is automatic evidence/selection, not a missing renderer. During narration the default source mix is a flat **0.04 linear gain, about −28 dB**, bounded at 0.15 (309–313), irrespective of whether the underlying source contains dialogue or an important effect. The flat bed cannot reproduce purposeful alternation by itself. Exact reference music/effects balances have not been auditioned.

Action: transcribe the clean source with timestamps, preserve uncertain speaker/name labels, and make eligible dialogue ranges available to planning. Let a reviewed storyboard choose narration versus an exact original-audio span, with no narrator overlapping that dialogue. Reserve its real duration before drafting the remaining voiceover. Reuse the existing dialogue renderer; do not fabricate dialogue from image descriptions.

Review target: a short V1-style reconstruction should retain one meaningful exchange and resume narration for the next causal action; V5's dialogue-led opening should remain possible. Audition the original and output turns, including lead-in/tail and competing music. Do not enforce a fixed dialogue quota across films.

## 2. Important actions and their framing are not verified through time

**Priority: high for fresh widescreen footage; mechanism gap certain, a particular output crop failure is not yet demonstrated.**

Reference evidence: V1 **0.00–6.54** shows the hand and cutting action before exposition; **8.30–11.94** moves to the severed digit/wedding ring; **17.50–21.96** shows the heated iron and reaction. The inspected first contact sheet includes hand/object inserts and close faces, not simply interchangeable action footage. These concrete images carry the narrated nouns and verbs. Reference visual sample times are approximate, with roughly one-second sampling uncertainty in V1. The originals are already square, so they cannot establish whether the creator added reframing or used an automated tracker.

Exact behavior: `planner.py::make_plan` describes each indexed span from **one midpoint image** and records the absence of continuous-motion analysis (263–272). `_describe` reduces that image to at most **384×384** (89–100). `_normalize_model_plan` passes source start/end without focus coordinates (217–219), and validation supplies `focus_x = 0.5` (172–179). `render.py::crop_filter` applies a constant horizontal offset plus vertical center crop for the entire shot (129–137); there is no focus_y, subject box, trajectory or in-shot action check. Multi-shot duration budgeting is being fixed separately and does not resolve these visibility limits.

Action: verify the beginning/action peak/end of chosen action spans and identify the story-critical person/object in the planned output crop. Store focus coordinates or crop keyframes when a centered square would lose them; allow static crops where they already work. Preserve a review image for the actual final crop at the action moment. A full motion model is not a prerequisite for this first correction.

Review target: on clean widescreen source footage, the opening hand/tool, ring, and iron must remain visible when their words are spoken. Check the rendered crop, rather than merely confirming a source-frame description mentions them. Do not assert the creator used any specific crop/tracking technology.

## 3. Captions match the broad look but omit observed opacity changes

**Priority: medium; static-versus-changing-opacity difference supported, exact reference animation unresolved.**

Reference evidence: V1 sampled frames around **2, 4, 6, 9, 11, 17, 21 and 23 seconds** contain partially visible/faded words or neighboring words of different brightness. Around **6–7 seconds**, “FIGHTING…” grows from a faint fragment to a readable phrase; around **9–11 seconds**, words in the wedding-ring phrase vary in visibility. This is consistent with timed fading or emphasis. Static one-second sheets do not establish the full animation, exact word onsets, font or easing. The compact white uppercase bold/slanted appearance at roughly **60–65%** image height is already broadly represented.

Exact behavior: `captions.py::_groups` groups up to seven words/48 characters and splits at a >0.35-second gap or sentence punctuation (34–49). `write_captions` makes one cue from first-word start to last-word end (67–70), writes the **whole phrase fully opaque**, uses a 32px nominal font on a 1080 square (74), and places every cue at `(width/2, height*0.60)` (84, 90–93). There are no ASS alpha, fade or word-dependent event overrides. Word timings are estimated by ASR, explicitly documented rather than forced alignment (96). Current placement and font size should not be changed speculatively.

Action: first inspect several original caption transitions at subsecond intervals, recording per-word appearance/disappearance and opacity. Add a configurable restrained fade/emphasis mode only from that evidence, retaining the existing plain version for comparison. Use actual audio timing; validate line breaks and legibility at phone size. Exact source timing and motion should remain adjustable rather than baked in as a presumed creator preset.

Review target: compare a 10–15-second clean-footage render with the V1 opening at phone size and normal playback. Text should remain compact, readable and close to the observed timing; identify any face/object obstruction in that concrete render. Do not treat the engineering preview's already-captioned reference footage as a production caption defect.

## Voice pace and sound: what can and cannot be concluded

`voice.py::_Synthesizer.generate` controls original Chatterbox exaggeration/CFG/temperature, gives Turbo/Nano no delivery kwargs, joins up-to-40-word chunks with a fixed 0.10-second gap, and has no explicit pace calibration (165–215). The three verified 13-word Nano clips are approximately **4.76 s Warguy / 4.44 s Kit Laser / 4.88 s Female**, or roughly **164 / 176 / 160 full-clip words/minute**. These tiny samples are not a reliable long-form narrator benchmark and have not been subjectively auditioned for likeness.

The references' **206–231 words/minute** measure **all mixed speech over the entire file**, including apparent original dialogue. It would be invalid to conclude the synthetic voices must all be accelerated by comparing those numbers directly. Before claiming fidelity, isolate and audition matched narration passages, compare articulation, pause placement and beat timing, then choose delivery or rewrite length from measured results. The completed full-output audio audition remains outstanding. The current −14 LUFS mastering is a reasonable controlled baseline, but the reference mixes themselves range substantially; copying their occasional positive true peaks would not establish a perceptual match.

Evidence and limits: `docs/reference-analysis.md`; `docs/reference-data/transcripts/V1.md` through V8; `/workspace/work/shorts-references/frames/V1/sheet-01.jpg`; the current planner/captions/voice/render source. No complete frame-by-frame viewing, isolated narrator rate analysis, full original/Turbo production run, voice-likeness audition, or fresh clean-film quality output is claimed here.
