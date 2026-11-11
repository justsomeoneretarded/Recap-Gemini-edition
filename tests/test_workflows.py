import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def _workflow(name):
    return yaml.load((ROOT/'.github/workflows'/name).read_text(),Loader=yaml.BaseLoader)


def test_short_workflow_is_separate_and_user_text_is_structured():
    workflow=_workflow('generate-shorts.yml')
    assert workflow['name']=='Generate Shorts'
    inputs=workflow['on']['workflow_dispatch']['inputs']
    assert len(inputs)<=10
    assert inputs['narrator']['options']==['warguy','kit-laser','female']
    assert inputs['aspect']['options']==['square','vertical']
    assert inputs['preset']['options'][0]=='gemini-balanced' and 'local-gemma4' in inputs['preset']['options']
    steps=workflow['jobs']['generate']['steps']
    names=[s.get('name') for s in steps]
    validation=names.index('Validate inputs before model downloads')
    models=names.index('Start Ollama and prepare local models')
    assert validation<names.index('Check the Gemini API key secret')<models
    assert steps[validation]['env']['SHORTS_INPUTS']=='${{ toJSON(inputs) }}'
    assert all('${{ inputs.' not in s.get('run','') for s in steps)
    assert steps[models]['if']=="env.SHORTS_NEEDS_OLLAMA == 'true'"
    assert 'pull_request' not in workflow['on']


def test_gemini_key_is_a_secret_scoped_to_the_steps_that_need_it():
    workflow=_workflow('generate-shorts.yml')
    text=(ROOT/'.github/workflows/generate-shorts.yml').read_text()
    inputs=workflow['on']['workflow_dispatch']['inputs']
    assert not [name for name in inputs if 'key' in name.lower() or 'secret' in name.lower()]
    assert 'GEMINI_API_KEY' not in json.dumps(workflow['jobs']['generate'].get('env',{}))
    users=[s['name'] for s in workflow['jobs']['generate']['steps'] if 'GEMINI_API_KEY' in s.get('env',{})]
    assert users==['Check the Gemini API key secret','Generate short']
    assert text.count('${{ secrets.GEMINI_API_KEY }}')==2 and 'echo "$GEMINI_API_KEY' not in text
    live=_workflow('gemini-live-check.yml')
    assert list(live['on'])==['workflow_dispatch']
    assert 'GEMINI' not in json.dumps(_workflow('ci.yml'))


def _inputs(**extra):
    return {'title':'T','source_url':'https://example.com/a.mp4','brief':'b','story_notes':'','narrator':'warguy',
            'target_seconds':'150','aspect':'square','preset':'gemini-balanced','source_window':'','options':'',**extra}


def test_workflow_options_map_to_validated_requests():
    from workflow_request import build
    request=build(_inputs(options='budget=$1.5 review_passes=3 analysis=detailed fallback=gemma4:12b speech=turbo',
                          source_window='51,73'))
    assert request['planner']=='gemini' and request['max_budget_usd']==1.5 and request['review_passes']==3
    assert request['analysis_quality']=='detailed' and request['fallback_writer']=='gemma4:12b'
    assert request['tts_model']=='turbo' and request['source_window']==[51.0,73.0]
    local=build(_inputs(preset='local-gemma4',options='writer=qwen3:4b-instruct-2507-q4_K_M'))
    assert local['planner']=='local' and local['script_model']=='qwen3:4b-instruct-2507-q4_K_M'
    for bad in ['colour=red','budget','review_passes=9','writer=gemma4:12b','analysis=8k']:
        with pytest.raises(ValueError):
            build(_inputs(options=bad))
