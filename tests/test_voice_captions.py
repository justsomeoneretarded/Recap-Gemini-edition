import json
import re
import subprocess
from pathlib import Path

import pytest

from shorts_pipeline.voice import _align_script, _chunks, synthesize_plan, validate_words, voice_reference
from shorts_pipeline import voice
from shorts_pipeline.captions import write_captions


def test_imported_audio_preserves_every_word_without_models(tmp_path):
    audio=tmp_path/'voice.wav'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=220:duration=2','-c:a','pcm_s16le',str(audio)],check=True)
    plan={'beats':[{'id':'b1','kind':'narration','text':'Hello there.','audio_path':str(audio),'words':[{'word':'Hello','start':.1,'end':.5},{'word':'there.','start':.6,'end':1.2}]}]}
    voiced=synthesize_plan(plan,'warguy',tmp_path)
    assert voiced['beats'][0]['duration']==pytest.approx(2,abs=.03)
    assert voiced['beats'][0]['timing_provenance']=='provided-word-timings'
    assert 'duration' not in plan['beats'][0]


def test_caption_offsets_and_script_punctuation(tmp_path):
    plan={'beats':[{'id':'b2','text':'Hello there.','duration':2,'timeline_start':4,'words':[{'word':'Hello','start':.1,'end':.5},{'word':'there.','start':.6,'end':1.2}]}]}
    files=write_captions(plan,tmp_path,1080,1080)
    assert '00:00:04,100 --> 00:00:05,200' in files['srt'].read_text()
    assert 'Hello there.' in files['srt'].read_text()
    assert r'\pos(540,648)' in files['ass'].read_text()
    assert 'HELLO THERE.' in re.sub(r'\{[^}]*\}', '', files['ass'].read_text())
    assert json.loads((tmp_path/'caption-timing.json').read_text())['beats'][0]['beat_id']=='b2'


@pytest.mark.parametrize('words',[
    [{'word':'Hello','start':.1,'end':.5}],
    [{'word':'Hello','start':.1,'end':1.2},{'word':'there.','start':.2,'end':.6}],
    [{'word':'Hello','start':.1,'end':.5},{'word':'there.','start':1.9,'end':3}],
])
def test_incomplete_or_invalid_word_timings_rejected(words):
    with pytest.raises(ValueError): validate_words('Hello there.',words,2)


def test_asr_cannot_silently_drop_script_words():
    with pytest.raises(ValueError):
        _align_script('He takes the bag and runs.',[{'word':word,'start':i*.2,'end':i*.2+.1} for i,word in enumerate('He takes bag and runs.'.split())],2)


def test_chunking_preserves_long_script():
    text=' '.join(f'word{i}' for i in range(125))
    pieces=_chunks(text)
    assert ' '.join(pieces)==text
    assert all(len(p.split())<=40 for p in pieces)


def test_dialogue_with_reviewed_timings_preserves_source_span(tmp_path):
    plan={'beats':[{'id':'d1','kind':'dialogue','text':'Go.','shots':[{'start':5,'end':8}],'words':[{'word':'Go.','start':.5,'end':1}]}]}
    voiced=synthesize_plan(plan,'female',tmp_path)
    assert voiced['beats'][0]['duration']==3
    assert voiced['beats'][0]['shots']==plan['beats'][0]['shots']


def test_explicit_reference_directory_has_no_silent_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('SHORTS_VOICE_DIR',str(tmp_path))
    with pytest.raises(FileNotFoundError): voice_reference('warguy')
    reference=tmp_path/'warguy.ogg'
    reference.write_bytes(b'fixture')
    assert voice_reference('warguy')==reference.resolve()


def test_asr_number_forms_preserve_written_script():
    recognized=[{'word':'He','start':0,'end':.2},{'word':'took','start':.2,'end':.4},
                {'word':'21','start':.4,'end':1.2},{'word':'coins.','start':1.2,'end':1.6}]
    aligned=_align_script('He took twenty one coins.',recognized,2)
    assert [w['word'] for w in aligned]==['He','took','twenty','one','coins.']
    assert aligned[2]['start']==pytest.approx(.4)
    assert aligned[3]['end']==pytest.approx(1.2)
    recognized=[{'word':'twenty','start':0,'end':.5},{'word':'one','start':.5,'end':1}]
    assert _align_script('21',recognized,1)[0]=={'word':'21','start':0,'end':1}


def _original_warguy_currency_transcript():
    # Actual original-Warguy pilot ASR uses '$14 million' for the intended
    # 'fourteen million dollar' phrase. Currency notation is an ASR estimate;
    # it does not independently certify whether the waveform spoke every word.
    text="Earlier, he broke the Continental hotel's rule. Now a fourteen million dollar bounty has made him a target for every assassin."
    tokens="Earlier, he broke the Continental Hotel's rule. Now a $14 million bounty has made him a target for every assassin.".split()
    starts=[0,.78,.84,1.02,1.2,1.54,2.12,3,3.12,3.34,3.62,4.04,4.56,4.78,4.94,5.08,5.14,5.42,5.64,5.94]
    ends=[.58,.84,1.02,1.2,1.54,2.12,2.4,3.12,3.34,3.62,4.04,4.56,4.78,4.94,5.08,5.14,5.42,5.64,5.94,6.28]
    return text,[{'word':word,'start':start,'end':end} for word,start,end in zip(tokens,starts,ends)]


def test_real_original_warguy_currency_notation_preserves_all_script_words():
    text,recognized=_original_warguy_currency_transcript()
    aligned=_align_script(text,recognized,6.64)
    assert [word['word'] for word in aligned]==text.split()
    assert aligned[9]=={'word':'fourteen','start':3.34,'end':3.62}
    assert aligned[10]['word']=='million'
    assert aligned[10]['start']==pytest.approx(3.62)
    assert aligned[10]['end']==pytest.approx(3.83)
    assert aligned[11]['word']=='dollar'
    assert aligned[11]['start']==pytest.approx(3.83)
    assert aligned[11]['end']==pytest.approx(4.04)
    assert validate_words(text,aligned,6.64)==aligned


def test_currency_word_omission_without_symbol_is_still_rejected():
    text,recognized=_original_warguy_currency_transcript()
    recognized[9]['word']='14'
    with pytest.raises(ValueError,match='could not find script words: dollar'):
        _align_script(text,recognized,6.64)


@pytest.mark.parametrize('text,recognized_text',[
    ('He won fourteen dollars.','He won $14.'),
    ('A fourteen dollar bounty.','A $14 bounty.'),
    ('He won one hundred fifty dollars.','He won $150.'),
    ('He won fourteen million dollars.','He won $14 million.'),
    ('A fourteen million dollar bounty.','A $14 million dollars bounty.'),
    ('A fourteen-million-dollar bounty.','A $14 million-dollar bounty.'),
    ('He won $14 million.','He won fourteen million dollars.'),
])
def test_usd_notation_and_explicit_currency_forms_share_estimated_intervals(text,recognized_text):
    recognized=[{'word':word,'start':index*.3,'end':(index+1)*.3}
                for index,word in enumerate(recognized_text.split())]
    aligned=_align_script(text,recognized,len(recognized)*.3)
    assert [word['word'] for word in aligned]==text.split()
    assert all(word['end']>word['start'] for word in aligned)
    assert aligned[-1]['end']==pytest.approx(recognized[-1]['end'])


@pytest.mark.parametrize('recover', [False,True])
def test_rejected_generated_speech_retains_bounded_audio_transcript_and_error(tmp_path,monkeypatch,recover):
    text='He takes the bag and runs.'
    omitted=[{'word':word,'start':index*.2,'end':index*.2+.15}
             for index,word in enumerate('He takes bag and runs.'.split())]
    complete=[{'word':word,'start':index*.2,'end':index*.2+.15}
              for index,word in enumerate(text.split())]
    reference=tmp_path/'warguy.ogg'
    reference.write_bytes(b'reference-fixture')
    monkeypatch.setattr(voice,'voice_reference',lambda *args: reference)
    monkeypatch.setattr(voice,'_duration',lambda *args: 2)
    calls=[]
    class Synthesizer:
        def __init__(self,*args): pass
        def generate(self,script,path,delivery,seed):
            calls.append(script)
            path.write_bytes(f'attempt-{len(calls)}'.encode())
    class Verifier:
        def transcribe(self,path):
            return complete if recover and len(calls)%2==0 else omitted
    monkeypatch.setattr(voice,'_Synthesizer',Synthesizer)
    monkeypatch.setattr(voice,'_Verifier',Verifier)
    plan={'beats':[{'id':'b1','kind':'narration','text':text}]}
    for invocation in range(2):
        if recover:
            result=synthesize_plan(plan,'warguy',tmp_path)
            assert result['beats'][0]['timing_provenance']=='asr-estimated-script-matched'
        else:
            with pytest.raises(RuntimeError,match='failed complete-speech verification'):
                synthesize_plan(plan,'warguy',tmp_path)
        diagnostics=tmp_path/'speech'/'diagnostics'
        errors=sorted(diagnostics.glob('*.json'))
        rejected=sorted(diagnostics.glob('*.wav'))
        assert len(errors)==len(rejected)==(1 if recover else 2)
        for error in errors:
            record=json.loads(error.read_text())
            assert record['text']==text
            assert record['recognized']==omitted
            assert record['duration']==2
            assert record['stage']=='alignment'
            assert record['error_type']=='ValueError'
            assert 'could not find script words' in record['error']
            assert Path(record['audio_path']).read_bytes().startswith(b'attempt-')
    assert len(calls)==(2 if recover else 4)
    # Failed attempts never acquire success metadata or masquerade as cache hits.
    assert len(list((tmp_path/'speech').glob('*.json')))==(1 if recover else 0)
