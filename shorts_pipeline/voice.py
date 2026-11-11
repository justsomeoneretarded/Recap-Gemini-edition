"""Local voice synthesis and honest, ASR-estimated caption alignment.

Heavy model libraries are imported only for synthesis/alignment. Reviewed audio
and word timings can be supplied in a storyboard without loading any model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

VOICES = ("warguy", "kit-laser", "female")
MODELS = ("original", "turbo", "nano")
VERSION = "shorts-speech-2"


def voice_reference(narrator: str) -> Path:
    """Locate references in an editable checkout or a packaged asset folder."""
    if narrator not in VOICES:
        raise ValueError("Unknown narrator.")
    configured=os.getenv("SHORTS_VOICE_DIR")
    if configured:
        candidates=[Path(configured).expanduser()/f"{narrator}.ogg"]
    else:
        package=Path(__file__).resolve().parent
        candidates=[package/"assets"/"voices"/f"{narrator}.ogg",
                    package.parent/"assets"/"voices"/f"{narrator}.ogg",
                    Path.cwd()/"assets"/"voices"/f"{narrator}.ogg"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing {narrator} reference. Set SHORTS_VOICE_DIR to a folder containing {narrator}.ogg.")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Media command failed: {result.stderr[-2000:]}")
    return result


def _duration(path: Path) -> float:
    data = json.loads(_run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]).stdout)
    if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
        raise ValueError(f"No audio stream in {path}")
    result = float(data["format"]["duration"])
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid audio duration for {path}")
    return result


def _normal(word: str) -> str:
    return re.sub(r"[^\w]", "", word.casefold(), flags=re.UNICODE)


def _integer_words(value: int) -> list[str]:
    small="zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    if value<20:
        return [small[value]]
    if value<100:
        tens="zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
        return [tens[value//10]]+(_integer_words(value%10) if value%10 else [])
    for scale,name in ((1000000000,"billion"),(1000000,"million"),(1000,"thousand"),(100,"hundred")):
        if value>=scale:
            return _integer_words(value//scale)+[name]+(_integer_words(value%scale) if value%scale else [])
    raise ValueError("Unsupported integer.")


def _lexical(word: str) -> list[str]:
    # Whisper may write a spoken number as digits. Treat those representations
    # equally without changing the written script. Decimal/ordinal ambiguity is
    # deliberately left to normal verification or reviewed timings.
    number=re.fullmatch(r"[\$£€]?([0-9][0-9,]*)(?:[.!?;:]?)",word)
    if number:
        value=int(number.group(1).replace(",",""))
        if value<1000000000000:
            return _integer_words(value)
    return ["dollar" if _normal(piece) == "dollars" else _normal(piece)
            for piece in re.split(r"[-–—]",word) if _normal(piece)]


def _contextual_units(words: list[str]) -> list[list[str]]:
    """Expand explicit USD notation without guessing an omitted currency unit.

    Whisper often writes spoken 'fourteen million dollars' as '$14 million'.
    Attribute the implicit unit to the final token of that numeric phrase; its
    recognized interval is later subdivided into ASR-estimated word intervals.
    Both script and ASR use this normalization, so written notation is preserved.
    """
    expanded = [_lexical(word) for word in words]
    magnitudes = {"hundred", "thousand", "million", "billion", "trillion"}
    for index, word in enumerate(words):
        if not re.fullmatch(r"\$[0-9][0-9,]*[.!?;:]?", word):
            continue
        last = index
        while last + 1 < len(words) and len(expanded[last + 1]) == 1 and expanded[last + 1][0] in magnitudes:
            last += 1
        following = expanded[last + 1] if last + 1 < len(words) else []
        if following and (following[0] == "dollar" or
                          (following[0] in magnitudes and following[-1] == "dollar")):
            continue  # An explicitly recognized dollar/dollars needs no insertion.
        expanded[last].append("dollar")
    return expanded


def validate_words(text: str, words: list[dict], duration: float) -> list[dict]:
    """Reject incomplete, unordered, or out-of-audio imported caption timings."""
    expected = [_normal(w) for w in text.split() if _normal(w)]
    actual = [_normal(str(w.get("word", ""))) for w in words if _normal(str(w.get("word", "")))]
    if expected != actual:
        raise ValueError("Word timings must contain every script word, in the same order.")
    previous = 0.0
    clean = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
            raise ValueError("Word timings require finite, positive intervals.")
        if start < previous - .04 or end > duration + .05:
            raise ValueError("Word timings overlap or extend beyond the audio.")
        previous = end
        clean.append({"word": str(word["word"]).strip(), "start": start, "end": min(end, duration)})
    if text.strip() and not clean:
        raise ValueError("Spoken text has no word timings.")
    return clean


def _align_script(text: str, recognized: list[dict], duration: float) -> list[dict]:
    script = [w for w in text.split() if _normal(w)]
    spoken = [w for w in recognized if _normal(w["word"])]
    if not script or not spoken:
        raise ValueError("Speech verification found no words.")
    script_units=[(unit,index) for index,units in enumerate(_contextual_units(script)) for unit in units]
    spoken_units=[]
    for word, units in zip(spoken, _contextual_units([word["word"] for word in spoken])):
        span=float(word["end"])-float(word["start"])
        for index,unit in enumerate(units):
            spoken_units.append({"unit":unit,"start":word["start"]+span*index/len(units),"end":word["start"]+span*(index+1)/len(units)})
    matcher = SequenceMatcher(None, [unit for unit,_ in script_units], [w["unit"] for w in spoken_units], autojunk=False)
    if matcher.ratio() < .86:
        raise ValueError("Generated speech does not sufficiently match the complete script.")
    matches = []
    for tag, i, j, a, b in matcher.get_opcodes():
        if tag == "delete":
            raise ValueError(f"Speech verification could not find script words: {' '.join(unit for unit,_ in script_units[i:j])}")
        if tag == "insert":
            if b - a > max(1, math.floor(len(script)*.03)):
                raise ValueError("Speech verification found extra spoken words.")
            continue
        if tag == "equal":
            matches.extend((script_units[k][1],spoken_units[a+k-i]) for k in range(i,j))
        else:
            # ASR commonly misspells names/numbers. Preserve the written script,
            # and explicitly classify these times as estimates, not forced alignment.
            if j-i != b-a:
                raise ValueError("ASR word counts differ; retry speech or provide reviewed word timings.")
            matches.extend((script_units[k][1],spoken_units[a+k-i]) for k in range(i,j))
    grouped={index:[] for index in range(len(script))}
    for index,word in matches:
        grouped[index].append(word)
    aligned=[{"word":word,"start":grouped[index][0]["start"],"end":grouped[index][-1]["end"]}
             for index,word in enumerate(script)]
    return validate_words(text, aligned, duration)


class _Verifier:
    def __init__(self):
        self.model = None

    def transcribe(self, path: Path) -> list[dict]:
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Speech verification requires faster-whisper; install the speech extra.") from exc
            self.model = WhisperModel(os.getenv("SHORTS_WHISPER_MODEL", "base.en"), device="cpu", compute_type="int8",
                                      cpu_threads=max(1, int(os.getenv("SHORTS_CPU_THREADS", "2"))))
        segments, _ = self.model.transcribe(str(path), language="en", beam_size=3, word_timestamps=True,
                                             vad_filter=True, condition_on_previous_text=False)
        return [{"word": w.word.strip(), "start": w.start, "end": w.end}
                for segment in segments for w in (segment.words or []) if w.word.strip()]


def _chunks(text: str, maximum: int = 40) -> list[str]:
    """Preserve every word while bounding original Chatterbox generations."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for sentence in sentences:
        words = sentence.split()
        while words:
            part, words = words[:maximum], words[maximum:]
            if result and len(result[-1].split()) + len(part) <= maximum:
                result[-1] += " " + " ".join(part)
            else:
                result.append(" ".join(part))
    return result


class _Synthesizer:
    def __init__(self, reference: Path, model: str):
        try:
            import torch
            import numpy as np
            import soundfile as sf
            if model == "original":
                from chatterbox.tts import ChatterboxTTS
                self.engine = ChatterboxTTS.from_pretrained(device="cpu")
            else:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.engine = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=model == "nano")
        except ImportError as exc:
            raise RuntimeError("Local narration requires the pinned Chatterbox speech environment. See README setup.") from exc
        self.torch, self.np, self.sf, self.name = torch, np, sf, model
        torch.set_num_threads(max(1, int(os.getenv("SHORTS_CPU_THREADS", "2"))))
        with torch.inference_mode():
            if model == "original":
                self.engine.prepare_conditionals(str(reference), exaggeration=.5)
            else:
                self.engine.prepare_conditionals(str(reference))

    def generate(self, text: str, path: Path, delivery: str, seed: int):
        settings = {"normal": (.5,.5,.8), "calm": (.4,.55,.75), "dramatic": (.6,.45,.8), "hype": (.65,.45,.8)}
        pieces=[]
        for i, chunk in enumerate(_chunks(text)):
            self.torch.manual_seed(seed+i)
            kwargs = dict(zip(("exaggeration","cfg_weight","temperature"),settings.get(delivery,settings["normal"]))) if self.name == "original" else {}
            with self.torch.inference_mode():
                samples=self.engine.generate(chunk, **kwargs).squeeze().detach().cpu().numpy()
            if not samples.size or not self.np.isfinite(samples).all() or float(self.np.max(self.np.abs(samples))) < 1e-5:
                raise ValueError("Narration model produced silent or invalid audio.")
            if pieces:
                pieces.append(self.np.zeros(round(.10*self.engine.sr),dtype=self.np.float32))
            pieces.append(samples)
        self.sf.write(str(path),self.np.concatenate(pieces),self.engine.sr,subtype="PCM_16")


def synthesize_plan(plan: dict, narrator: str, workdir: Path, model: str = "original") -> dict:
    if narrator not in VOICES or model not in MODELS:
        raise ValueError("Choose Warguy, Kit Laser or Female and original, turbo or nano.")
    result=copy.deepcopy(plan)
    cache=Path(workdir)/"speech"
    cache.mkdir(parents=True,exist_ok=True)
    reference=None
    verifier=_Verifier()
    engine=None
    for index, beat in enumerate(result.get("beats",[])):
        text=str(beat.get("text","")).strip()
        kind=beat.get("kind","narration")
        if kind not in ("narration","dialogue"):
            raise ValueError("Unknown beat kind.")
        if kind == "dialogue":
            if len(beat.get("shots",[])) != 1:
                raise ValueError("A retained dialogue beat needs exactly one source shot.")
            shot=beat["shots"][0]
            duration=float(shot["end"])-float(shot["start"])
            if duration<=0:
                raise ValueError("Invalid dialogue interval.")
            if beat.get("words"):
                beat["words"]=validate_words(text,beat["words"],duration)
                beat.setdefault("timing_provenance","provided-word-timings")
            else:
                audio=cache/f"dialogue-{index}.wav"
                _run(["ffmpeg","-y","-v","error","-threads","1","-ss",str(shot["start"]),"-i",str(result["source_path"]),"-t",str(duration),"-vn","-ac","1","-ar","16000",str(audio)])
                spoken=verifier.transcribe(audio)
                if not text:
                    text=" ".join(w["word"] for w in spoken)
                    beat["text"]=text
                beat["words"]=_align_script(text,spoken,duration)
                beat["timing_provenance"]="asr-estimated-source-dialogue"
            beat["duration"]=duration
            continue
        if not text:
            raise ValueError("A narration beat cannot have empty text.")
        if beat.get("audio_path"):
            audio=Path(beat["audio_path"]).resolve()
            duration=_duration(audio)
            if not beat.get("words"):
                beat["words"]=_align_script(text,verifier.transcribe(audio),duration)
                beat["timing_provenance"]="asr-estimated-imported-audio"
            else:
                beat["words"]=validate_words(text,beat["words"],duration)
                beat.setdefault("timing_provenance","provided-word-timings")
            beat.update(audio_path=str(audio),duration=duration)
            continue
        if reference is None:
            reference=voice_reference(narrator)
        fingerprint=hashlib.sha256(json.dumps([VERSION,text,narrator,model,beat.get("delivery","normal"),hashlib.sha256(reference.read_bytes()).hexdigest()],ensure_ascii=False).encode()).hexdigest()
        audio=cache/f"{fingerprint}.wav"
        metadata=cache/f"{fingerprint}.json"
        if audio.exists() and metadata.exists():
            saved=json.loads(metadata.read_text())
            duration=_duration(audio)
            words=validate_words(text,saved["words"],duration)
        else:
            if engine is None:
                engine=_Synthesizer(reference,model)
            error=None
            for attempt in range(2):
                spoken=None
                duration=None
                stage="generation"
                try:
                    engine.generate(text,audio,beat.get("delivery","normal"),int(fingerprint[:8],16)+attempt*101)
                    stage="duration"
                    duration=_duration(audio)
                    stage="transcription"
                    spoken=verifier.transcribe(audio)
                    stage="alignment"
                    words=_align_script(text,spoken,duration)
                    metadata.write_text(json.dumps({"words":words,"duration":duration,"model":model,"narrator":narrator,"timing_provenance":"asr-estimated-script-matched"},indent=2))
                    error=None
                    break
                except (RuntimeError, ValueError) as exc:
                    error=exc
                    # The two deterministic attempt slots bound retained evidence
                    # for this beat/text. A retry never erases the failed waveform.
                    diagnostics=cache/"diagnostics"
                    diagnostics.mkdir(exist_ok=True)
                    stem=diagnostics/f"beat-{index:04d}-{fingerprint}-attempt-{attempt+1}"
                    rejected=stem.with_suffix(".wav")
                    if audio.exists():
                        audio.replace(rejected)
                    else:
                        rejected.unlink(missing_ok=True)  # Never link an older attempt's waveform.
                    stem.with_suffix(".json").write_text(json.dumps({
                        "version":VERSION,"beat_id":beat.get("id",index),"attempt":attempt+1,
                        "text":text,"model":model,"narrator":narrator,"stage":stage,
                        "duration":duration,"recognized":spoken,"error":str(exc),
                        "error_type":type(exc).__name__,
                        "audio_path":str(rejected.resolve()) if rejected.exists() else None,
                        "timing_provenance":"asr-estimated-script-matching-attempt",
                    },indent=2),encoding="utf-8")
            if error:
                raise RuntimeError(f"Narration beat {beat.get('id',index)} failed complete-speech verification: {error}") from error
        beat.update(audio_path=str(audio.resolve()),duration=duration,words=words,timing_provenance="asr-estimated-script-matched")
    result["narrator"]=narrator
    result["tts_model"]=model
    return result
