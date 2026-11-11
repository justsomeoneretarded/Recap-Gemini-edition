"""Compact, word-timed captions for square and vertical recap videos."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .voice import validate_words


def _srt_time(value: float) -> str:
    millis=round(value*1000)
    hours,millis=divmod(millis,3600000)
    minutes,millis=divmod(millis,60000)
    seconds,millis=divmod(millis,1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _ass_time(value: float) -> str:
    ticks=round(value*100)
    hours,ticks=divmod(ticks,360000)
    minutes,ticks=divmod(ticks,6000)
    seconds,ticks=divmod(ticks,100)
    return f"{hours}:{minutes:02}:{seconds:02}.{ticks:02}"


def _wrap(words: list[str]) -> str:
    if len(words) <= 1 or len(" ".join(words)) <= 27:
        return " ".join(words)
    midpoint=min(range(1,len(words)),key=lambda i:abs(len(" ".join(words[:i]))-len(" ".join(words[i:]))))
    return " ".join(words[:midpoint])+"\n"+" ".join(words[midpoint:])


def _groups(words: list[dict]) -> list[list[dict]]:
    groups=[]
    current=[]
    for word in words:
        length=len(" ".join([w["word"] for w in current]+[word["word"]]))
        gap=float(word["start"])-float(current[-1]["end"]) if current else 0
        if current and (len(current)>=7 or length>48 or gap>.35):
            groups.append(current)
            current=[]
        current.append(word)
        if word["word"].endswith((".","!","?",";")):
            groups.append(current)
            current=[]
    if current:
        groups.append(current)
    return groups


def _escape_ass(text: str) -> str:
    return text.upper().replace("\\","\\\\").replace("{","\\{").replace("}","\\}").replace("\n",r"\N")


def _word_fade_text(words: list[dict], text: str, cue_start: float, cue_end: float, beat_start: float) -> tuple[str, bool]:
    """Keep the full phrase laid out while revealing its words on their timings.

    This approximates the sampled reference's moving reveal with word alpha,
    rather than claiming to reproduce its unknown character-mask animation.
    Very short cues stay fully visible so the animation cannot hide their text.
    """
    duration=round((cue_end-cue_start)*1000)
    starts=[max(0,round((beat_start+float(word["start"])-cue_start)*1000)) for word in words]
    if duration<300 or duration-starts[-1]<80:
        return _escape_ass(text),False
    first_line_words=len(text.split("\n",1)[0].split()) if "\n" in text else None
    parts=[]
    for index,(word,entrance) in enumerate(zip(words,starts)):
        available=duration-entrance
        fade=min(120,available//3)
        visible=min(60,available//3)
        # Earlier words leave first, but every word gets time at full opacity.
        exit_start=max(entrance+fade+visible,duration-fade-(len(words)-1-index)*30)
        exit_end=min(duration,exit_start+fade)
        if index:
            parts.append(r"\N" if index==first_line_words else " ")
        parts.append(
            f"{{\\alpha&HFF&\\t({entrance},{entrance+fade},\\alpha&H00&)"
            f"\\t({exit_start},{exit_end},\\alpha&HFF&)}}{_escape_ass(word['word'])}"
        )
    return "".join(parts),True


def write_captions(plan: dict, outdir: Path, width: int, height: int) -> dict:
    if width<=0 or height<=0:
        raise ValueError("Caption canvas dimensions must be positive.")
    style=plan.get("caption_style","reference_word_fade")
    if style not in {"reference_word_fade","static"}:
        raise ValueError("caption_style must be 'reference_word_fade' or 'static'.")
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    cues=[]
    provenance=[]
    previous_end=0.0
    for beat in plan.get("beats",[]):
        start=float(beat["timeline_start"])
        duration=float(beat["duration"])
        if not all(map(math.isfinite,(start,duration))) or start<0 or duration<=0 or start<previous_end-.04:
            raise ValueError("Caption timeline must have positive, ordered beat intervals.")
        words=validate_words(str(beat.get("text","")),beat.get("words",[]),duration)
        provenance.append({"beat_id":beat.get("id"),"timing":beat.get("timing_provenance","provided-word-timings")})
        for group in _groups(words):
            cue_start=start+group[0]["start"]
            cue_end=min(start+duration,start+group[-1]["end"])
            cues.append((cue_start,cue_end,_wrap([w["word"] for w in group]),group,start,start+duration))
        previous_end=start+duration
    srt=outdir/"captions.srt"
    srt.write_text("\n\n".join(f"{i}\n{_srt_time(cue[0])} --> {_srt_time(cue[1])}\n{cue[2]}" for i,cue in enumerate(cues,1))+"\n",encoding="utf-8")
    size=max(12,round(min(width,height)*.0296))
    header=f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Recap,DejaVu Sans,{size},&H00FFFFFF,&H00FFFFFF,&H60000000,&H80000000,-1,-1,0,0,100,100,1,0,1,0.6,1.2,5,50,50,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines=[]
    short_static_cues=0
    for index,(start,end,text,words,beat_start,beat_end) in enumerate(cues):
        if style=="reference_word_fade":
            next_start=cues[index+1][0] if index+1<len(cues) else beat_end
            end=min(end+.1,beat_end,next_start)
            clean,animated=_word_fade_text(words,text,start,end,beat_start)
            short_static_cues+=not animated
        else:
            clean=_escape_ass(text)
        position=f"{{\\an5\\pos({width//2},{round(height*.60)})}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Recap,,0,0,0,,{position}{clean}")
    ass=outdir/"captions.ass"
    ass.write_text(header+"\n".join(lines)+"\n",encoding="utf-8")
    (outdir/"caption-timing.json").write_text(json.dumps({
        "note":"Whisper timing is an ASR estimate, not phonetic forced alignment. Supplied timings are validated structurally, not acoustically.",
        "caption_style":style,
        "animation":{
            "implementation":"Custom word-alpha approximation of sampled reference reveal/fade; the original creator's mask algorithm is unknown." if style=="reference_word_fade" else "Static full-phrase captions.",
            "word_timing":"Provided or aligned word starts drive the reveal; full phrase layout remains fixed.",
            "max_fade_ms":120 if style=="reference_word_fade" else 0,
            "max_ass_hold_ms":100 if style=="reference_word_fade" else 0,
            "short_cues_rendered_static":short_static_cues,
            "srt":"Original phrase text and word-boundary intervals, without animation or extra hold."
        },
        "beats":provenance
    },indent=2))
    return {"srt":srt,"ass":ass}
