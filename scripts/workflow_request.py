"""Read structured Actions inputs without interpolating user text into shell code."""
import json
import os
import hashlib
from datetime import datetime,timezone
from pathlib import Path
from shorts_pipeline.config import DEFAULTS, LOCAL_WRITERS, request_config

PRESETS = {'gemini-free':('gemini','free'),'gemini-balanced':('gemini','balanced'),'gemini-quality':('gemini','quality'),
           'gemini-economy':('gemini','economy'),'local-gemma4':('local',None)}
# Advanced options: space- or comma-separated key=value pairs in one text box.
OPTION_KEYS = {'budget':'max_budget_usd','review_passes':'review_passes','analysis':'analysis_quality',
               'fallback':'fallback_writer','speech':'tts_model','alignment':'forced_alignment',
               'strict':'strict_review','writer':'script_model','rpm':'requests_per_minute',
               'max_requests':'max_requests'}


def parse_options(text: str) -> dict:
    result = {}
    for token in text.replace(',', ' ').split():
        if '=' not in token:
            raise ValueError(f'Option {token!r} must look like key=value. Keys: {", ".join(OPTION_KEYS)}.')
        key, value = token.split('=', 1)
        if key not in OPTION_KEYS:
            raise ValueError(f'Unknown option {key!r}. Use one of: {", ".join(OPTION_KEYS)}.')
        name = OPTION_KEYS[key]
        if name == 'max_budget_usd':
            value = float(value.lstrip('$'))
        elif name == 'requests_per_minute':
            value = float(value)
        elif name in ('review_passes', 'max_requests'):
            value = int(value)
        elif name == 'strict_review':
            value = value.lower() in ('1', 'true', 'yes')
        result[name] = value
    return result


def build(data: dict) -> dict:
    window = json.loads('['+data['source_window']+']') if data.get('source_window','').strip() else None
    planner, preset = PRESETS[data.get('preset') or 'gemini-free']
    request = {'title':data['title'], 'source':data['source_url'], 'brief':data['brief'],
               'story_notes':data.get('story_notes',''), 'narrator':data['narrator'],
               'target_seconds':float(data['target_seconds']), 'aspect':data['aspect'],
               'source_window':window, 'planner':planner, **parse_options(data.get('options',''))}
    if preset:
        request['gemini_preset'] = preset
        if 'script_model' in request:
            raise ValueError('writer= applies to the local-gemma4 preset; with Gemini presets use fallback=.')
    request.setdefault('script_model', DEFAULTS['script_model'])
    if request['script_model'] not in LOCAL_WRITERS:
        raise ValueError(f'writer must be one of {", ".join(LOCAL_WRITERS)}.')
    if preset is None:
        # The local preset is the original Gemma pipeline; record it explicitly.
        request.pop('max_budget_usd', None)
    return request_config(request)


if __name__ == '__main__':
    data = json.loads(os.environ['SHORTS_INPUTS'])
    request = build(data)
    Path('work').mkdir(exist_ok=True)
    Path('work/request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    needs_ollama = request['planner']=='local' or request['fallback_writer']!='none'
    writer = request['script_model'] if request['planner']=='local' else request['fallback_writer']
    print(json.dumps({k:request[k] for k in ('planner','gemini_preset','fallback_writer','tts_model','target_seconds')
                      if k in request}))
    if os.environ.get('GITHUB_ENV'):
        source_code=hashlib.sha256(Path('shorts_pipeline/sources.py').read_bytes()).hexdigest()
        key=hashlib.sha256(json.dumps([request['source'],request['source_window'],source_code,
                                     datetime.now(timezone.utc).date().isoformat()]).encode()).hexdigest()
        with open(os.environ['GITHUB_ENV'],'a',encoding='utf-8') as stream:
            stream.write(f'SHORTS_SOURCE_CACHE_KEY={key}\n')
            stream.write(f'SHORTS_PLANNER={request["planner"]}\n')
            stream.write(f'SHORTS_NEEDS_OLLAMA={"true" if needs_ollama else "false"}\n')
            stream.write(f'SCRIPT_MODEL={writer if needs_ollama else DEFAULTS["script_model"]}\n')
