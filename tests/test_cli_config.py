import json
import re
import subprocess
from pathlib import Path
import pytest
from shorts_pipeline.config import request_config, assign_timeline
from shorts_pipeline.cli import generate, main


def test_independent_short_inputs_and_measured_budget():
    request = request_config({'title':'A short','source':'footage.mp4'})
    assert request['aspect']=='square'
    assert request['narrator']=='warguy'
    for window in [[1,0],[float('nan'),4],[0,1,2]]:
        with pytest.raises(ValueError):
            request_config({'title':'A','source':'x','source_window':window})
    for invalid in [float('nan'),181,0]:
        with pytest.raises(ValueError):
            request_config({'title':'A','source':'x','target_seconds':invalid})
    with pytest.raises(ValueError,match='audio will not be cut'):
        assign_timeline({'beats':[{'duration':16}]},15)
    plan=assign_timeline({'beats':[{'duration':1.25},{'duration':2.5}]},15)
    assert plan['beats'][1]['timeline_start']==1.25
    assert plan['duration']==3.75


def test_complete_cli_path_with_reviewed_audio(tmp_path):
    # Synthetic media exercises pipeline wiring, not model voice quality.
    source=tmp_path/'source.mp4'
    audio=tmp_path/'tone.wav'
    def ff(*args):
        subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-filter_threads','1',*map(str,args)],check=True,timeout=30)
    ff('-f','lavfi','-i','testsrc2=size=160x160:rate=30:duration=2',
       '-c:v','libx264','-threads','1',source)
    ff('-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1.5',audio)
    storyboard=tmp_path/'plan.json'
    storyboard.write_text(json.dumps({'beats':[{'text':'Reviewed fixture.', 'audio_path':'tone.wav',
        'words':[{'word':'Reviewed','start':.1,'end':.6},{'word':'fixture.','start':.7,'end':1.3}],
        'shots':[{'start':0,'end':2}]}]}))
    out=tmp_path/'out'
    result=generate({'title':'Synthetic pipeline test','source':str(source),'plan':str(storyboard),
                     'target_seconds':15,'fps':30,'preview_width':160,'source_window':[0,2]},tmp_path/'work',out)
    assert Path(result['video']).is_file()
    assert json.loads((out/'status.json').read_text())['state']=='succeeded'
    assert 'REVIEWED FIXTURE.' in re.sub(r'\{[^}]*\}','',(out/'captions.ass').read_text())
    assert (out/'script.txt').read_text()=='Reviewed fixture.'
    assert len(list(out.glob('cover-*.jpg')))==3
    assert json.loads((out/'source-attribution.json').read_text())['source_sha256']
    assert json.loads((out/'source-attribution.json').read_text())['original_source_window']==[0,2]
    with pytest.raises(FileExistsError,match='fresh --output'):
        generate({'title':'Another run','source':str(source)},tmp_path/'work',out)
    assert json.loads((out/'status.json').read_text())['state']=='succeeded'


def test_failure_records_stage_and_never_claims_success(tmp_path):
    out=tmp_path/'out'
    with pytest.raises(FileNotFoundError):
        generate({'title':'Missing source','source':str(tmp_path/'missing.mp4')},tmp_path/'work',out)
    status=json.loads((out/'status.json').read_text())
    assert status['state']=='failed' and status['stage']=='source'
    assert not (out/'final.mp4').exists()


def test_automatic_path_keeps_source_speech_through_planning_and_render(tmp_path,monkeypatch):
    # Synthetic tones verify stage wiring; the recognition data is mocked.
    from shorts_pipeline import sources,source_speech,planner
    source=tmp_path/'synthetic.mp4';voice=tmp_path/'narration.wav'
    def ff(*args):
        subprocess.run(['ffmpeg','-y','-v','error','-threads','1','-filter_threads','1',*map(str,args)],check=True,timeout=30)
    ff('-f','lavfi','-i','testsrc2=size=160x160:rate=30:duration=4',
       '-f','lavfi','-i','sine=frequency=250:sample_rate=48000:duration=4',
       '-c:v','libx264','-threads','1','-c:a','aac','-shortest',source)
    ff('-f','lavfi','-i','sine=frequency=700:sample_rate=48000:duration=1.5',voice)
    candidate={'id':'D0001','start':2.,'end':3.,'text':'Hold on.',
               'words':[{'word':'Hold','start':.1,'end':.4},{'word':'on.','start':.5,'end':.9}],
               'confidence':.95,'timing_provenance':'asr-estimated-source-speech'}
    monkeypatch.setattr(source_speech,'analyze_source_speech',lambda p,w:{'candidates':[candidate],'language':'en'})
    monkeypatch.setattr(sources,'index_source',lambda p,w:[{'id':'S1','start':0,'end':4}])
    def make(request,path,scenes,work,*,source_speech):
        assert source_speech==[candidate]
        return {'beats':[
            {'id':'N1','kind':'narration','text':'Reviewed fixture.','audio_path':str(voice),
             'words':[{'word':'Reviewed','start':.1,'end':.6},{'word':'fixture.','start':.7,'end':1.3}],
             'shots':[{'start':0,'end':2}]},
            {'id':'D1','kind':'dialogue','text':candidate['text'],'words':candidate['words'],
             'timing_provenance':candidate['timing_provenance'],'shots':[{'start':2,'end':3}]}]}
    monkeypatch.setattr(planner,'make_plan',make)
    out=tmp_path/'out'
    result=generate({'title':'Synthetic automatic-stage validation','source':str(source),
                     'target_seconds':15,'fps':30,'preview_width':160},tmp_path/'work',out)
    timeline=json.loads((out/'timeline.json').read_text())
    assert result['qc']['status']=='passed'
    assert timeline['beats'][1]['kind']=='dialogue'
    assert timeline['beats'][1]['text']=='Hold on.'
    assert timeline['beats'][1]['duration']==1.
    assert timeline['beats'][1]['timing_provenance']=='asr-estimated-source-speech'
    assert json.loads((out/'source-speech.json').read_text())['candidates']==[candidate]
