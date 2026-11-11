# Eight reference Shorts: evidence and production guide

These exports consistently prioritize a specific action, a visible object and a developing danger before explaining the wider plot. They combine compact narration with selected source-dialogue passages, keep movie imagery filling a square canvas, and use restrained white captions. The shared method is a useful production profile; there is no evidence here that any individual technique caused retention or channel growth.

## Scope and confidence

The source of truth is the eight downloaded release assets listed in `assets.json`, with source IDs V1–V8, asset IDs, sizes, URLs and SHA-256 hashes. This report combines:

- **Measured:** media probes; full-file FFmpeg scene-change and full-mix EBU R128 measurements; V1 sampled border and opening-frame repetition measurements.
- **Automatic ASR:** complete `transcripts/V1.md` through `V8.md` and their segment/word JSON. These were generated with `faster-whisper small.en`, CPU int8, English. Words, names, speaker changes and timestamps require correction against the sound before becoming production subtitles or verified film facts.
- **Sampled visual observation:** contact sheets at nominal one-second spacing for V1 and four-second spacing for V2–V8. The inspection reliably establishes representative layouts and story objects, not every cut, frame, transition or expression. V1 opening and ending samples received closer inspection; cross-video visual comparisons emphasize the first contact sheet of each file.
- **Interpretation:** story structure and recommendations derived from the preceding evidence. Proposed implementation defaults are explicitly distinguished from measured settings.

No complete frame-by-frame human viewing, subjective audio audition, source-film alignment, speaker diarization, exact font identification or narrator-voice identification was performed. Contact-sheet labels are nominal sample indexes, not verified frame timestamps: FFmpeg's `fps` rounding can select a frame from within a sampling window. Visual times should therefore be treated as approximate, with up to roughly the sampling interval of uncertainty. Narrative timestamps below come from automatic ASR and should be checked during final editorial review.

## Source inventory and measured envelope

| ID | Reference title | Duration | Automatic transcript words / full-minute rate | Change candidates >0.25 / median interval | Full-mix LUFS / true peak |
|---|---|---:|---:|---:|---:|
| V1 | Everyone Wants Him Dead | 136.60 s | 469 / 206.0 | 50 / 2.13 s | −14.0 / −5.5 dBFS |
| V2 | He's Not a Cop at All | 178.21 s | 665 / 223.9 | 43 / 2.17 s | −14.0 / −5.1 dBFS |
| V3 | He Was Hired to Kill. Now He's the Prey | 159.31 s | 554 / 208.7 | 69 / 1.48 s | −7.7 / +1.2 dBFS |
| V4 | He Was the Target All Along | 148.74 s | 570 / 229.9 | 63 / 2.05 s | −8.1 / +0.5 dBFS |
| V5 | One Man vs the Deadliest Fighter | 135.69 s | 489 / 216.2 | 49 / 2.13 s | −7.7 / +0.5 dBFS |
| V6 | Rookie Cop Just Learned the Hard Truth | 179.53 s | 657 / 219.6 | 25 / 2.54 s | −8.5 / +0.8 dBFS |
| V7 | The Next Assassin Is Already on His Way—but This Father Is Ready | 153.85 s | 593 / 231.3 | 24 / 2.87 s | −8.6 / +0.4 dBFS |
| V8 | The Quiet Guy Was the Deadliest | 169.43 s | 622 / 220.3 | 81 / 1.48 s | −7.8 / +0.7 dBFS |

All eight supplied streams are **1080×1080, H.264, reported 60 fps**, with **44.1 kHz stereo AAC** audio. A title containing “4K” does not change the measured 1080-square delivery. Container frame rate does not establish the original film rate, interpolation method or unique-frame rate. V1's opening ten-second comparison found only one exactly identical adjacent grayscale pair out of 599; it does not identify the source or conversion process.

The word-rate column counts all transcribed speech, including apparent movie dialogue, against full file duration. It is **not isolated narrator words per minute**. The reproducible count uses word-like tokens including apostrophes; ASR errors affect the result. It is a rough density range for planning, not a requirement to accelerate every synthetic voice to 231 words/minute.

Scene-change candidates are an FFmpeg image-difference measurement on 270×270 frames, not a reviewed shot list. Counts above 0.12 range from 55 to 126; counts above 0.25 range from 24 to 81. Fast action, flashes and lighting changes can trigger candidates, and similar-looking cuts can be missed. Use the timestamps to suggest review points, not to force a cut every two seconds. Detailed commands, candidate scores and limitations are in `measurements/`.

## V1: detailed narrative and editing map

The reference first shows an extreme price being paid, then jumps backward to explain why the protagonist needs protection. That ordering creates a concrete question before supplying background. The later narrative follows a deadline, with dialogue taking over at selected confrontations.

| ASR time | What the transcript says / sampled image evidence | Structural function |
|---|---|---|
| 0.00–6.54 | Palm on table, chisel, severed ring finger. Opening sampled images show hands and the small cutting object prominently. | Immediate irreversible action; no greeting or film-title introduction. |
| 6.82–17.24 | Pain, severed digit, wedding ring from his late wife; offering buys protection from the assassin underworld. | Converts the shocking physical event into emotional cost and an immediate motive. |
| 17.50–21.96 | Elder retrieves hot iron and cauterizes the wound. | Completes the opening action sequence through a second concrete image. |
| 22.36–23.78 | “I wish you good luck on your path.” | Short apparent source-dialogue beat after narration; speaker attribution is an inference pending audition. |
| 24.24–31.04 | He rises, enters a tent and finds a bulletproof suit. | Brief consequence/reward for the opening bargain. |
| 31.38–43.80 | John is introduced as a top assassin, explains the hotel rule and the $14 million bounty. | **Delayed backstory starts about 31 seconds in.** The reference earns its explanation with a prior action rather than beginning with biography. |
| 44.00–50.84 | Winston gives a one-hour head start. | Establishes the time constraint. |
| 51.04–67.68 | Wounded flight through streets; every passerby may be a killer; last coin pays a driver to take the dog to safety. | Threat becomes spatial and personal. The dog/coin task gives the chase a small, understandable objective. |
| 68.20–72.70 | Concierge request, then a response transcribed as “Yes, sir, Mr. Wythe.” | Apparent source conversation; the name is visibly an ASR reliability concern and should not be adopted as a fact. |
| 73.10–82.24 | Library stash, then a giant blocks his path before the deadline. | New obstacle interrupts a practical survival task. |
| 82.86–92.34 | “I still have time”; disputed deadline; “Who's gonna know the difference?”; money exchange. ASR merges speakers in the final segment. | The longest early dialogue-led confrontation gives the antagonist a motive without narrator exposition. Exact wording/speaker turns need correction. |
| 93.26–97.86 | Narration summarizes the giant's defeat. | Compresses the fight payoff and returns to forward motion. |
| 98.68–105.48 | Private doctor risks punishment to treat him. | Temporary refuge with a rule-based limit. |
| 106.58–114.94 | Clock strikes six; doctor must stop; John stitches himself. | Pays off the deadline through a visible consequence rather than merely repeating the bounty. |
| 115.26–120.30 | Doctor asks to be shot in non-lethal places to protect his cover. | A surprising practical reversal; retain character logic, not just violence. |
| 121.38–126.56 | “Good luck, Mr. Wick”; later “Thanks, Doc.” | Dialogue creates a release before the next escalation. The gap is not proof of silence: untranscribed speech, effects or music may remain. |
| 126.98–136.44 | Assassins emerge again because the bounty remains active. Last sampled images continue with street/assassin footage and captions. | Ends on renewed danger. There is no generic spoken subscribe prompt in the ASR and no end card in the inspected final samples. |

V1's effect depends on **event selection and ordering**, not simply a fast voice. Every major passage changes the situation: sacrifice → temporary protection → reason for pursuit → deadline → obstacle → refuge → deadline expires → renewed pursuit. The opening precedes earlier causal history; a production planner must support an intentional cold open followed by a clear chronological reset.

## Cross-reference patterns and counterexamples

| ID | Opening and delayed explanation, using ASR times | Main escalation | Ending behavior |
|---|---|---|---|
| V1 | 0.00–6.54 self-injury; assassin/bounty context at 31.38–43.80 | Deadline makes help expire; apparent dialogue at 68.20–72.70 and 82.86–92.34 | 126.98–136.44 renewed citywide threat |
| V2 | 0.00–6.60 unusual skin-scrubbing preparation; target's weekly cash at 27.04–34.52 | Parallel roles, guard suspicion, vault access, courier disguise and ambulance escape | 160.20–177.90 opens the next bank robbery rather than closing with a moral |
| V3 | 0.00–8.18 five locks and upside-down cup; rich-assassin/poor-flat contradiction at 8.40–13.14 | Secret room, concealed weapon, paid job becomes a trap | 146.49–152.41 unexpected second arrival; last ASR is about 6.9 seconds before file end, so the ending needs sound review |
| V4 | 0.00–6.42 nearly kills girlfriend; relationship suspicion before the final-job setup | A gunmaker unknowingly supplies the weapon aimed at him; later explanation reverses that setup | 142.24–148.48 wounded reunion and reflection; **not a required next-episode cliffhanger** |
| V5 | 0.00–6.98 apparent original dialogue about an amulet; narrator explains fighters/revenge at 7.28–17.58 | Clearly differentiated rounds: physical mismatch, glass-wrapped fists, weapons, memory-driven comeback | 116.20–135.42 victory, revenge fulfilled, goodbye/new life; **complete payoff** |
| V6 | First recognized speech at 5.97; corrupt-police confrontation and rescued girl precede a broader moral argument | Dialogue-led conflict about policing, then a routine stop becomes an armed crisis | 173.56–179.34 rookie reaches a new understanding; **thematic payoff** |
| V7 | 0.00–5.98 man avoids cameras; accidental exposure at 6.34–14.68, identity/hunt at 14.98–32.16 | Domestic warning, island landing, traps, daughter endangered, surviving response | 139.52–153.60 first squad defeated, stronger assassin dispatched |
| V8 | 0.00–13.60 excellent shooter pretends to be weak; son's death motivates job at 13.84–28.68 | Concealment → lethal reveal → robber backstory → gang recognition → police knowledge | 159.04–169.10 unresolved identity/dramatic irony rather than a conventional victory card |

What repeats:

1. **A concrete behavior precedes a dossier.** A chisel, brush, cup, camera, amulet or rifle can carry the first question. Open with a meaningful event that can be shown, then add the minimum needed cause. V5 starts with apparent character dialogue and V6 has several seconds before recognized narration, so “narrator must speak at 0.00” is not a universal rule.
2. **Narration operates in cause-and-effect steps.** Characters act, the situation changes, a constraint appears, and the next action answers it. The references frequently delay an explanation until the audience has seen the surprising consequence: V3 hidden weapon and V4 rigged rifle are strong examples. Do not delay information so far that the viewer cannot follow the immediate goal.
3. **The camera can illustrate specific nouns and verbs.** Representative frames show V1 hands/ring/suit; V2 preparation and security equipment; V3 pipe/hidden room/armory; V4 rifle parts and faces; V5 amulet, glass wraps and weapons; V6 gloves and patrol encounters; V7 cameras, searchlights and traps; V8 firing range, cash bags and armored transport. A generic pool of action footage would lose this relationship.
4. **Original dialogue is an editorial tool.** V1 has short greetings and a longer standoff; V4 has a technical client conversation at roughly 27–42 seconds; V5 begins in dialogue; V6 has a substantial argument around 83–104 seconds. Preserve selected source dialogue and give it room in the mix. Do not blindly cover every second with narration. ASR alone cannot certify speaker identity or the exact division of source speech and voiceover.
5. **Escalation has a shape.** V5 uses three fighting phases; V7 uses detection, defense, personal loss, counterattack and a next threat; V2 uses preparation, suspicion, access and exit. Build distinct beats rather than increasing adjective intensity while repeating the same situation.
6. **Endings vary with the story.** Threat, ironic reveal, reflection, thematic realization and full resolution all occur. The pipeline should select and validate an ending mode; it should not append the same cliffhanger or call to action to every script.

## Image, captions and editing observations

The representative frames use full-bleed square footage, including close faces, hands and story objects alongside occasional wider geography. Captions are compact white uppercase text with a bold, slanted appearance, predominantly one or two lines, centered horizontally in the central-lower image region, approximately 60–65% of image height in the inspected samples. This is a visual estimate from reduced-size frames, **not a measured font or exact baseline specification**. Some samples show faded neighboring words beside brighter words. A later [10-fps filmstrip of V1 from nominal 1.8–4.6 seconds](reference-data/caption-motion/v1-reveal-fade.jpg) confirms progressive reveal and fade in that passage; [sampling details](reference-data/caption-motion/sampling.json) preserve its scope. The creator's exact font, character mask and easing remain unresolved, and this passage does not establish identical animation throughout all eight videos.

The reference appearance does not support oversized colored text covering half the picture. An initial implementation can use a licensed bold italic font, white fill, a restrained dark shadow, compact phrase groups and an adjustable baseline; validate readability at phone size and against actual faces/objects. Match caption timing to the final synthesized or retained speech, rather than stretching transcript text across guessed durations.

Shot/reverse-shot faces, object inserts and large changes in composition are visible. The samples do not show a recurring elaborate transition template, and detector candidates suggest frequent changes, but sampling cannot exclude short transitions, speed changes or effects. Do not claim a universal cut duration. Shots should begin and end to communicate the action, its reaction or the next line's subject. Avoid cutting away before the narrated action becomes legible.

The supplied scenes vary in color: cool/warm contrasts in V1, warmer interiors in V3/V5, cooler night/island material in V7. There is insufficient evidence for one global grade, sharpening model, interpolation tool, upscaler or crop algorithm. V1 has no persistent near-black border in the measured samples. Recovering the exact film crop or deciding whether a camera movement was added requires comparison with the original footage.

## A practical production profile

These are **initial engineering/editorial defaults derived from evidence**, not claims about the creator's internal tools.

| Control | Starting behavior | Basis / qualification |
|---|---|---|
| Output | Explicit `reference_square` profile: 1080×1080; 135–180-second observed range; reported 60-fps delivery available | All eight exports. A vertical profile can coexist but should be chosen explicitly. Check actual target-platform duration/aspect support separately. |
| Story selection | One protagonist or objective with visible stakes; choose a manageable continuous mini-arc | Cross-eight interpretation; not a promise that every film can be compressed equally well. |
| Hook | Start with the strongest understandable action, contradiction or meaningful dialogue; attach exact footage | V1/V3/V7/V8; dialogue-led V5 is a counterexample to fixed narrator templates. |
| Script plan | Cold-open option, clear reset, causal beats, selected dialogue slots, named ending mode | V1 ordering and varied endings. Every claim should have a source span. |
| Speech budget | Use roughly 200–230 mixed-speech words/minute only as a planning signal; reserve time for dialogue and action | ASR counts are 206–231 over full duration. Render speech first and revise to measured duration. |
| Shot assembly | Semantic scene matching; retain action/reaction and important objects; variable cut length | Candidate-change medians vary 1.48–2.87 seconds and are not real shot counts. |
| Captions | Compact uppercase white, bold italic appearance, usually 1–2 lines, adjustable baseline around y=0.62 | Sampled visual estimate; use actual audio alignment and test phone readability. |
| Sound | Separate narration, retained source audio, music and effects; duck competing material when dialogue matters | Dialogue presence inferred from ASR. Exact source balance needs audition. |
| Loudness | Start with −14 LUFS integrated and a true-peak ceiling at or below −1.5 dBTP, then audition | A deliberate clean-output policy; only V1/V2 measure −14. The louder references are not a reason to reproduce positive true-peak overs. |
| Ending | Explicit `renewed_threat`, `reveal`, `reflection`, `thematic_payoff` or `resolved_payoff` | Cross-eight counterexamples. No mandatory generic subscribe/end card. |
| Voice | Select an available configured preset and audition it; retain vendor/model provenance | Warguy, Kitlaser and Female presets were retrieved from the user's Top10 repository; see the [model and voice review](top10-model-review.md). Their availability does not identify the reference creator's narrator, vendor or TTS model. |

## Pipeline acceptance checks

The first implementation should produce a reviewable package: source inventory, scene index, evidence-linked outline, timed script with dialogue slots, generated narration, edit decision list, captions, final render and QA results. A knowledge graph is useful for querying these artifacts across references; it does not replace the media or prove observations missing from the evidence.

Before accepting a rendered Short:

- Confirm the selected aspect ratio, resolution, real output duration and playable audio/video streams against the chosen profile.
- For every story beat, preserve source ID, original time span and the sentence/claim it supports. A generated line without a source span must be flagged for review.
- Keep a deliberate hook and explain the protagonist's immediate goal early enough to follow the sequence. Preserve chronological resets explicitly, especially for a V1-style cold open.
- Check that the narrated action is actually visible and occurs while that line is heard. Review hands, faces, weapons, text and other story-critical objects after reframing.
- Preserve chosen movie dialogue without accidental competing narration; verify speaker turns and names against the sound. A gap in ASR is not automatically empty audio.
- Generate captions from the final audio/alignment, check word correctness, line breaks, readable size, bounds and visibility at phone scale. Caption positions must adapt when an essential face/object would be obscured.
- Measure final mix loudness and true peak; then audition intelligibility, abrupt volume changes, clip-boundary clicks and competing music. Passing an EBU meter alone does not establish a good mix.
- Require an actual ending beat and no accidental blank tail, truncated sentence or forced universal cliffhanger.
- Validate the whole first render continuously. Contact sheets and automatic metrics are triage tools, not a substitute for this final playback review.

V1 is a strong first reconstruction target because it tests the cold open, delayed backstory, deadline, dialogue interludes, meaningful objects and a renewed-threat ending. V5 and V6 should then be held-out editorial checks: they challenge a pipeline that overfits V1 by requiring a resolved fight or a thematic change rather than another assassin cliffhanger.

## Data quality and open questions

Automatic recognition contains errors and merged turns. V1's “Mr. Wythe” at 71.50–72.70 and “He's almost up” at 83.70–84.32 are warning examples; do not silently convert uncertain text into a verified quote. V3 has an untranscribed final tail of roughly 6.9 seconds. V5's apparent dialogue near the end of round one is not reliably represented by ASR. V6's recognized speech begins at 5.97 seconds; that does not prove the opening was silent.

The references themselves may contain inaccurate or compressed plot claims. V4 calls a person dead at 70.68–74.84 and barely alive at 75.10–83.90. V8's narrated robbery statistics around 28.68–39.18 are arithmetically inconsistent. These are reasons to verify story facts and source actions, rather than cloning every transcript detail. Technical weapon explanations and character opinions are claims within the reference, not independently validated facts or recommendations.

We do not yet have a verified exact narrator/TTS identity, original soundtrack separation, the creator's editing timeline, complete source-film versions, or a direct comparison between original movie and final crop. No evidence here supports an exact thumbnail-generation method, sourcing service, script model or visual-generation model. Those choices must be assessed through repository configuration and controlled output examples. A graph can preserve these unknowns and their provenance; it should not fill them with invented certainty.
