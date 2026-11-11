"""Prepare the configured local Ollama models; fail when setup is incomplete."""
import os
import time
import requests
from shorts_pipeline.config import DEFAULTS

url = os.environ.get('OLLAMA_URL','http://127.0.0.1:11434').rstrip('/')
deadline = time.monotonic()+60
while True:
    try:
        response=requests.get(url+'/api/tags',timeout=3)
        response.raise_for_status()
        break
    except requests.RequestException:
        if time.monotonic()>=deadline:
            raise RuntimeError('Ollama did not start; inspect the uploaded model logs.')
        time.sleep(1)
for model in [os.environ.get('SCRIPT_MODEL',DEFAULTS['script_model']),os.environ.get('SCENE_MODEL',DEFAULTS['vision_model'])]:
    print(f'Preparing {model}',flush=True)
    with requests.post(url+'/api/pull',json={'model':model,'stream':True},stream=True,timeout=(10,1800)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                import json
                state=json.loads(line)
                if state.get('error'):
                    raise RuntimeError(f'{model}: {state["error"]}')
                if state.get('status')=='success':
                    print(f'{model} ready',flush=True)
