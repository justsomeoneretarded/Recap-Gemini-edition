from __future__ import annotations
import math

DEFAULTS = {'brief': '', 'story_notes': '', 'narrator': 'warguy', 'tts_model': 'original',
            'target_seconds': 150, 'aspect': 'square', 'fps': 60,
            'script_model': 'gemma4:12b', 'vision_model': 'qwen3-vl:2b-instruct-q4_K_M',
            'ollama_url': 'http://127.0.0.1:11434', 'max_scenes': 120,
            'auto_source_speech': True, 'caption_style': 'reference_word_fade',
            'planner': 'gemini', 'gemini_preset': 'free', 'fallback_writer': 'none',
            'forced_alignment': 'off', 'script_mode': 'hybrid'}
LOCAL_WRITERS = ('gemma4:12b', 'qwen3:4b-instruct-2507-q4_K_M')


def request_config(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError('Request must be a JSON object.')
    result = {**DEFAULTS, **data}
    for name in ('title', 'source'):
        if not isinstance(result.get(name), str) or not result[name].strip():
            raise ValueError(f'Provide a nonempty {name}.')
    for name in ('brief', 'story_notes', 'script_model', 'vision_model', 'ollama_url'):
        if not isinstance(result[name], str):
            raise ValueError(f'{name} must be text.')
    for name, choices in [('narrator', ('warguy','kit-laser','female')),
                          ('tts_model', ('original','turbo','nano')), ('aspect', ('square','vertical'))]:
        if result[name] not in choices:
            raise ValueError(f'{name} must be one of {", ".join(choices)}.')
    if not isinstance(result['auto_source_speech'],bool):
        raise ValueError('auto_source_speech must be true or false.')
    if result['caption_style'] not in ('reference_word_fade','static'):
        raise ValueError('caption_style must be reference_word_fade or static.')
    duration = float(result['target_seconds'])
    if not math.isfinite(duration) or not 15 <= duration <= 180:
        raise ValueError('target_seconds must be between 15 and 180.')
    result['target_seconds'] = duration
    if result.get('source_window') is not None:
        window=result['source_window']
        if not isinstance(window,(list,tuple)) or len(window)!=2:
            raise ValueError('source_window must be [start_seconds,end_seconds].')
        start,end=map(float,window)
        if not all(map(math.isfinite,(start,end))) or not 0<=start<end:
            raise ValueError('source_window must contain finite increasing nonnegative seconds.')
        result['source_window']=[start,end]
    if result['planner'] not in ('gemini','local'):
        raise ValueError('planner must be gemini or local.')
    if result['fallback_writer'] not in ('none',)+LOCAL_WRITERS:
        raise ValueError(f'fallback_writer must be none or one of {", ".join(LOCAL_WRITERS)}.')
    if result['script_mode'] not in ('hybrid','story'):
        raise ValueError('script_mode must be hybrid or story.')
    if result['forced_alignment'] not in ('off','auto'):
        raise ValueError('forced_alignment must be off or auto.')
    if result['planner']=='gemini':
        from .gemini.settings import settings_from_request
        settings_from_request(result)  # Validate presets, models, passes and budget before any work.
    if result['fps'] not in (30,60):
        raise ValueError('fps must be 30 or 60.')
    maximum = result['max_scenes']
    if not isinstance(maximum,int) or isinstance(maximum,bool) or not 1 <= maximum <= 1000:
        raise ValueError('max_scenes must be an integer between 1 and 1000.')
    return result


def assign_timeline(plan: dict, target_seconds: float) -> dict:
    cursor = 0.0
    beats = plan.get('beats', [])
    if not beats:
        raise ValueError('Storyboard has no beats.')
    for beat in beats:
        duration = float(beat['duration'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'Beat {beat.get("id")} has an invalid measured duration.')
        beat['timeline_start'] = cursor
        cursor += duration
        beat['timeline_end'] = cursor
    if cursor > min(180, float(target_seconds)) + .10:
        raise ValueError(f'Measured final duration {cursor:.2f}s exceeds the {target_seconds:g}s budget. Shorten the script; audio will not be cut.')
    plan['duration'] = cursor
    return plan
