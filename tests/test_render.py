"""Small synthetic media tests. Fixtures contain tones, not generated narration."""
from pathlib import Path
from array import array
import shutil
import subprocess

import pytest

from shorts_pipeline.render import _probe, allocate_shots, crop_filter, render_plan


def test_coverage_never_loops_or_stretches_unreasonably():
    with pytest.raises(ValueError, match="add source shots"):
        allocate_shots([{"start": 0, "end": 1}], 3, 10, 30)
    shots = allocate_shots([{"start": 0, "end": 2}], 2.2, 10, 30)
    assert .85 <= shots[0]["speed"] <= 1
    assert shots[0]["end"] <= 2
    assert sum(s["frames"] for s in shots) == 66


def test_invalid_source_range_and_focus_are_rejected():
    for shot in ({"start": -1, "end": 2}, {"start": 0, "end": 12}, {"start": 0, "end": 2, "focus_x": "nan"}):
        with pytest.raises(ValueError):
            allocate_shots([shot], 1, 10, 30)
    assert "*0.200000" in crop_filter(1080, 1920, .2)
    with pytest.raises(ValueError):
        crop_filter(1079, 1080)


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "1", "-filter_threads", "1", *map(str,args)], check=True,timeout=30)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_full_render_preserves_dialogue_and_supplied_audio(tmp_path):
    source = tmp_path/"synthetic-source.mp4"
    voice = tmp_path/"synthetic-tone.wav"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4", "-f", "lavfi", "-i", "sine=frequency=250:sample_rate=48000:duration=4",
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", source)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000:duration=1.5", voice)
    plan = {"title": "Synthetic integration fixture", "beats": [
        {"id": "n1", "kind": "narration", "text": "Synthetic test tone.", "audio_path": str(voice), "duration": 1.5,
         "timeline_start": 0, "shots": [{"start": 0, "end": 1}, {"start": 1, "end": 2}]},
        {"id": "d1", "kind": "dialogue", "text": "", "duration": 1, "timeline_start": 1.5, "shots": [{"start": 2, "end": 3}]}]}
    out = tmp_path/"output"
    out.mkdir()
    (out/"captions.ass").write_text("""[Script Info]
ScriptType: v4.00+
PlayResX: 180
PlayResY: 320
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,15,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,-1,0,0,100,100,0,0,1,1,1,5,5,5,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,SYNTHETIC TEST
""", encoding="utf-8")
    result = render_plan(plan, source, out, width=180, height=320, fps=30)
    assert result["video"].is_file()
    assert result["qc"]["status"] == "passed"
    assert abs(result["qc"]["duration_seconds"]-2.5) < .18
    assert result["qc"]["width"] == 180
    assert result["qc"]["height"] == 320
    assert len(result["covers"]) == 3
    assert result["contact_sheet"].is_file()
    assert result["provenance"].is_file()
    assert result["qc"]["captions_burned"]
    # Check that supplied narration dominates its section and the original
    # source tone returns during the retained-dialogue section.
    def frequency(at):
        data = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(result["video"]),
                                        "-t", "0.3", "-vn", "-f", "s16le", "-ac", "1", "-ar", "8000", "-"])
        samples = array("h", data)
        crossings = sum(a < 0 <= b for a,b in zip(samples, samples[1:]))
        return crossings/.3
    assert 650 < frequency(.5) < 750
    assert 220 < frequency(1.8) < 280


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_narration_audio_mismatch_fails_before_render(tmp_path):
    source, voice = tmp_path/"source.mp4", tmp_path/"voice.wav"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x160:rate=30:duration=1", "-c:v", "libx264", "-threads", "1", source)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=500:duration=1", voice)
    plan = {"beats": [{"kind": "narration", "audio_path": str(voice), "duration": 4, "shots": [{"start": 0, "end": 1}]}]}
    with pytest.raises(ValueError, match="does not match"):
        render_plan(plan, source, tmp_path/"out", width=160, height=160, fps=30)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
@pytest.mark.parametrize("shot_start", [0, 1])
def test_short_or_absent_source_bed_never_truncates_narration(tmp_path, shot_start):
    source, voice = tmp_path/"sparse-source.mp4", tmp_path/"voice.wav"
    # Four seconds of video, but only half a second of original source sound.
    # The second parameter selects footage entirely after that sound has ended.
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x160:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=250:sample_rate=48000:duration=0.5",
            "-c:v", "libx264", "-threads", "1", "-c:a", "aac", source)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000:duration=2", voice)
    plan = {"beats": [{"id": "n1", "kind": "narration", "audio_path": str(voice), "duration": 2,
                       "shots": [{"start": shot_start, "end": shot_start+2}]}]}
    result = render_plan(plan, source, tmp_path/"out", width=160, height=160, fps=30)
    assert _probe(result["video"])["audio_end"] >= 1.99
    assert result["qc"]["audio_end_seconds"] >= result["qc"]["last_voice_endpoint_seconds"]-.02
    late_audio = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(result["video"]),
                                         "-t", "0.3", "-vn", "-f", "s16le", "-ac", "1", "-ar", "8000", "-"])
    samples = array("h", late_audio)
    assert len(samples) >= 2300
    crossings = sum(a < 0 <= b for a,b in zip(samples, samples[1:]))
    assert 650 < crossings/.3 < 750  # The late narration survives, not silence.


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_retained_dialogue_rejects_missing_source_audio(tmp_path):
    source = tmp_path/"sparse-source.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x160:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=250:sample_rate=48000:duration=0.5",
            "-c:v", "libx264", "-threads", "1", "-c:a", "aac", source)
    plan = {"beats": [{"kind": "dialogue", "duration": 2, "shots": [{"start": 0, "end": 2}]}]}
    with pytest.raises(ValueError, match="missing source audio"):
        render_plan(plan, source, tmp_path/"out", width=160, height=160, fps=30)


def test_excess_footage_keeps_every_production_anchor_and_midpoint():
    # Production first beat: the suit appears only in the fourth anchor.
    lengths = [2.1, 1.183, 1.133, 1.183]
    selected = [{"start": 10*i, "end": 10*i+length, "focus_x": .2+i*.1, "shot_id": f"S{i+1}"}
                for i, length in enumerate(lengths)]
    allocated = allocate_shots(selected, 4, 40, 60)
    assert len(allocated) == 4
    assert [s["shot_id"] for s in allocated] == ["S1", "S2", "S3", "S4"]
    assert sum(s["frames"] for s in allocated) == 240
    for original, actual in zip(selected, allocated):
        assert actual["frames"] >= 15
        assert actual["speed"] == 1
        assert actual["focus_x"] == original["focus_x"]
        assert original["start"] <= actual["start"] < actual["end"] <= original["end"]
        assert actual["start"]+actual["end"] == pytest.approx(original["start"]+original["end"])
        assert actual["source_start"] == original["start"]
        assert actual["source_end"] == original["end"]


@pytest.mark.parametrize("mode, expected", [("start", (1, 2)), ("center", (2, 3)), ("end", (3, 4))])
def test_reviewed_trim_modes(mode, expected):
    shot = allocate_shots([{"start": 1, "end": 4, "trim_mode": mode}], 1, 5, 30)[0]
    assert (shot["start"], shot["end"]) == expected
    with pytest.raises(ValueError, match="trim_mode"):
        allocate_shots([{"start": 0, "end": 1, "trim_mode": "other"}], .5, 5, 30)


def test_anchor_budget_rejects_illegible_cuts_and_caps_short_ranges():
    with pytest.raises(ValueError, match="Too many selected shots"):
        allocate_shots([{"start": i, "end": i+1} for i in range(5)], 1, 5, 30)
    shots = allocate_shots([{"start": 0, "end": .1}, {"start": 1, "end": 2}], .5, 3, 30)
    assert [s["frames"] for s in shots] == [3, 12]
    assert sum(s["frames"] for s in shots) == 15
    full = allocate_shots([{"start": 0, "end": 1}, {"start": 2, "end": 3}], 2.2, 4, 30)
    assert [(s["start"], s["end"]) for s in full] == [(0, 1), (2, 3)]
    assert all(.85 <= s["speed"] <= 1 for s in full)
    assert sum(s["frames"] for s in full) == 66


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is required")
def test_short_narration_render_retains_all_four_visual_anchors(tmp_path):
    source, voice = tmp_path/"colors.mp4", tmp_path/"voice.wav"
    colors = ["0xFF0000", "0x00FF00", "0x0000FF", "0xFFFF00"]
    inputs = []
    for color in colors:
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=160x160:r=30:d=1"]
    _ffmpeg(*inputs, "-filter_complex_threads", "1", "-filter_complex",
            "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]", "-map", "[v]",
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", source)
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=700:sample_rate=48000:duration=2", voice)
    plan = {"beats": [{"kind": "narration", "audio_path": str(voice), "duration": 2,
                       "shots": [{"start": i, "end": i+1} for i in range(4)]}]}
    result = render_plan(plan, source, tmp_path/"out", width=160, height=160, fps=30)
    assert result["qc"]["status"] == "passed"
    # Sample the actual encoded video, not merely the allocated plan. Old
    # tail-trimming rendered red and green only, silently losing blue/yellow.
    for at, expected in zip([.25, .75, 1.25, 1.75], [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        rgb = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "1", "-ss", str(at),
                                       "-i", str(result["video"]), "-vf", "scale=1:1", "-frames:v", "1",
                                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        assert len(rgb) == 3
        assert all(abs(actual-target) < 35 for actual, target in zip(rgb, expected))


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is required')
def test_short_looping_music_is_bounded_by_the_actual_edit(tmp_path):
    source=tmp_path/'silent-video.mp4'
    voice=tmp_path/'voice-tone.wav'
    music=tmp_path/'short-music.wav'
    _ffmpeg('-f','lavfi','-i','color=c=blue:s=160x160:r=30:d=2',
            '-c:v','libx264','-threads','1',source)
    _ffmpeg('-f','lavfi','-i','sine=frequency=700:sample_rate=48000:duration=1.2',voice)
    _ffmpeg('-f','lavfi','-i','sine=frequency=220:sample_rate=48000:duration=0.25',music)
    plan={'beats':[{'id':'N','kind':'narration','text':'Synthetic audio fixture.',
                   'audio_path':str(voice),'duration':1.2,'timeline_start':0,
                   'shots':[{'start':0,'end':2}]}]}
    result=render_plan(plan,source,tmp_path/'output',160,160,30,music)
    assert result['qc']['status']=='passed'
    assert 1.15<=result['qc']['duration_seconds']<=1.35
    assert result['qc']['audio_end_seconds']>=1.15
    data=subprocess.check_output(['ffmpeg','-v','error','-ss','0.7','-i',str(result['video']),
                                  '-t','0.3','-vn','-f','s16le','-ac','1','-ar','8000','-'],timeout=30)
    samples=array('h',data)
    crossings=sum(a<0<=b for a,b in zip(samples,samples[1:]))/.3
    assert 650<crossings<750  # The supplied voice remains dominant after music repeats.
    # Endpoint metadata alone can pass after silence-padding a truncated voice.
    # Verify the actual last portion of the supplied tone remains audible too.
    tail=subprocess.check_output(['ffmpeg','-v','error','-ss','1.10','-i',str(result['video']),
                                  '-t','0.08','-vn','-f','s16le','-ac','1','-ar','8000','-'],timeout=30)
    end_samples=array('h',tail)
    assert len(end_samples)>=600
    assert sum(abs(sample) for sample in end_samples)/len(end_samples)>500
    tail_frequency=sum(a<0<=b for a,b in zip(end_samples,end_samples[1:]))/(len(end_samples)/8000)
    assert 650<tail_frequency<750
