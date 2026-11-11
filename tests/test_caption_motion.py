import json
import subprocess

import pytest

from shorts_pipeline.captions import write_captions


def _plan(text="Alpha beta", words=None, duration=2.2, start=0):
    return {"beats":[{
        "id":"b1","text":text,"timeline_start":start,"duration":duration,
        "words":words or [
            {"word":"Alpha","start":.1,"end":.7},
            {"word":"beta","start":.9,"end":1.8},
        ],
    }]}


def _events(ass):
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def test_word_fades_use_actual_starts_with_fixed_white_phrase_layout(tmp_path):
    files=write_captions(_plan(start=4),tmp_path,1080,1080)
    ass=files["ass"].read_text()
    event=_events(ass)[0]
    assert "0:00:04.10,0:00:05.90" in event
    assert r"\t(0,120,\alpha&H00&)" in event
    assert r"\t(800,920,\alpha&H00&)" in event
    assert "ALPHA" in event and "BETA" in event
    assert r"\pos(540,648)" in event
    assert "&H00FFFFFF,&H00FFFFFF" in ass
    assert ",-1,-1,0,0," in ass  # White, bold italic; no colored karaoke.
    assert r"\k" not in event
    assert files["srt"].read_text()=="1\n00:00:04,100 --> 00:00:05,800\nAlpha beta\n"
    provenance=json.loads((tmp_path/"caption-timing.json").read_text())
    assert provenance["caption_style"]=="reference_word_fade"
    assert "approximation" in provenance["animation"]["implementation"]
    assert "unknown" in provenance["animation"]["implementation"]


def test_ass_hold_stops_at_next_cue_and_beat_end(tmp_path):
    words=[{"word":"Go.","start":.1,"end":.5},{"word":"Now.","start":.55,"end":1}]
    files=write_captions(_plan("Go. Now.",words,duration=1),tmp_path,1080,1080)
    events=_events(files["ass"].read_text())
    assert "0:00:00.10,0:00:00.55" in events[0]
    assert "0:00:00.55,0:00:01.00" in events[1]
    assert "00:00:00,100 --> 00:00:00,500" in files["srt"].read_text()


def test_short_cues_remain_readable_without_hidden_words(tmp_path):
    files=write_captions(_plan("Go.",[{"word":"Go.","start":0,"end":.12}],duration=.12),tmp_path,1080,1080)
    event=_events(files["ass"].read_text())[0]
    assert r"\alpha" not in event
    assert event.endswith("GO.")
    provenance=json.loads((tmp_path/"caption-timing.json").read_text())
    assert provenance["animation"]["short_cues_rendered_static"]==1


def test_static_style_keeps_phrase_text_and_original_interval(tmp_path):
    plan=_plan()
    plan["caption_style"]="static"
    files=write_captions(plan,tmp_path,1080,1080)
    event=_events(files["ass"].read_text())[0]
    assert "0:00:00.10,0:00:01.80" in event
    assert event.endswith("ALPHA BETA")
    assert r"\alpha" not in event


def test_animation_preserves_literal_escaping_and_two_lines(tmp_path):
    text=r"Grab {the} back\slash with brutal force."
    words=[{"word":word,"start":index*.25,"end":index*.25+.2} for index,word in enumerate(text.split())]
    files=write_captions(_plan(text,words,duration=2),tmp_path,1080,1080)
    event=_events(files["ass"].read_text())[0]
    assert event.count(r"\N")==1
    assert r"\{THE\}" in event
    assert r"BACK\\SLASH" in event
    assert "{the}" in files["srt"].read_text()
    assert r"back\slash" in files["srt"].read_text()
    assert files["srt"].read_text().splitlines()[2:]==["Grab {the} back\\slash","with brutal force."]


def test_unknown_caption_style_is_rejected(tmp_path):
    plan=_plan()
    plan["caption_style"]="karaoke"
    with pytest.raises(ValueError,match="caption_style"):
        write_captions(plan,tmp_path,1080,1080)


def test_ffmpeg_renders_more_words_later_without_recentring(tmp_path):
    """Exercise libass pixels: a tag-only regression would miss broken transforms."""
    write_captions(_plan(),tmp_path,640,640)
    result=subprocess.run([
        "ffmpeg","-v","error","-filter_threads","1","-f","lavfi","-i",
        "color=c=black:s=640x640:r=10:d=2",
        "-vf",r"ass=captions.ass,select=eq(n\,3)+eq(n\,13)+eq(n\,18)",
        "-vsync","0","-frames:v","3","-threads","1","-f","rawvideo","-pix_fmt","gray","pipe:1",
    ],cwd=tmp_path,check=True,capture_output=True,timeout=30)
    frame_size=640*640
    assert len(result.stdout)==frame_size*3
    early=result.stdout[:frame_size]
    later=result.stdout[frame_size:frame_size*2]
    leaving=result.stdout[frame_size*2:]
    early_pixels=[index for index,value in enumerate(early) if value>150]
    later_pixels=[index for index,value in enumerate(later) if value>150]
    leaving_pixels=[index for index,value in enumerate(leaving) if value>150]
    assert len(early_pixels)>100
    assert len(later_pixels)>len(early_pixels)*1.4
    assert min(index%640 for index in early_pixels)==min(index%640 for index in later_pixels)
    assert max(index%640 for index in later_pixels)>max(index%640 for index in early_pixels)+20
    assert 0<len(leaving_pixels)<len(later_pixels)*.7
    assert min(index%640 for index in leaving_pixels)>min(index%640 for index in early_pixels)+20
