from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_short_workflow_is_separate_and_user_text_is_structured():
    path=ROOT/'.github/workflows/generate-shorts.yml'
    workflow=yaml.load(path.read_text(),Loader=yaml.BaseLoader)
    assert workflow['name']=='Generate Shorts'
    inputs=workflow['on']['workflow_dispatch']['inputs']
    assert len(inputs)<=10
    assert inputs['narrator']['options']==['warguy','kit-laser','female']
    steps=workflow['jobs']['generate']['steps']
    validation=next(i for i,s in enumerate(steps) if s.get('name')=='Validate inputs before model downloads')
    models=next(i for i,s in enumerate(steps) if s.get('name')=='Start Ollama and prepare models')
    assert validation<models
    assert steps[validation]['env']['SHORTS_INPUTS']=='${{ toJSON(inputs) }}'
    assert all('${{ inputs.' not in s.get('run','') for s in steps)
    assert 'pull_request' not in workflow['on']
