from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from .config import request_config, assign_timeline
from .media import probe, write_json, excerpt


def generate(request: dict, workdir: Path, outdir: Path) -> dict:
    from .sources import resolve_source, index_source
    from .planner import make_plan, validate_storyboard
    from .voice import synthesize_plan
    from .captions import write_captions
    from .render import render_plan
    request = request_config(request)
    workdir, outdir = Path(workdir).resolve(), Path(outdir).resolve()
    workdir.mkdir(parents=True,exist_ok=True)
    outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'final.mp4').exists():
        raise FileExistsError('Output folder already contains final.mp4. Choose a fresh --output folder to preserve the previous completed run.')
    started = time.time()
    status = {'state':'running','stage':'source','started_at_epoch':started}
    def stage(name):
        status['stage'] = name
        write_json(outdir/'status.json',status)
        print(f'Shorts: {name}',flush=True)
    try:
        stage('source')
        source = resolve_source(request['source'],workdir)
        if request.get('source_window'):
            stage('source excerpt')
            source=excerpt(source,workdir,request['source_window'])
        source_info = probe(source)
        if request.get('plan'):
            plan_path = Path(request['plan']).resolve()
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            # Imported media paths are relative to the storyboard file.
            for beat in plan.get('beats',[]):
                if beat.get('audio_path') and not Path(beat['audio_path']).is_absolute():
                    beat['audio_path'] = str((plan_path.parent/beat['audio_path']).resolve())
            plan = validate_storyboard(plan,source_info['duration'])
            plan['planning_mode'] = 'supplied-storyboard'
        else:
            candidates=[]
            if request['auto_source_speech'] and source_info['has_audio']:
                from .source_speech import analyze_source_speech
                stage('source speech')
                speech=analyze_source_speech(source,workdir)
                write_json(outdir/'source-speech.json',speech)
                candidates=speech['candidates']
            stage('index')
            scenes = index_source(source,workdir)
            stage('storyboard')
            plan = make_plan(request,source,scenes,workdir,source_speech=candidates)
        plan['title'] = request['title']
        plan['source_path'] = str(source)
        plan['source_origin'] = request['source']
        plan['source_window'] = request.get('source_window')
        plan['caption_style'] = request['caption_style']
        write_json(outdir/'request.json',request)
        write_json(outdir/'storyboard.json',plan)
        stage('narration')
        plan = synthesize_plan(plan,request['narrator'],workdir,model=request['tts_model'])
        plan = assign_timeline(plan,request['target_seconds'])
        write_json(outdir/'timeline.json',plan)
        width,height = (1080,1080) if request['aspect']=='square' else (1080,1920)
        # CLI-only small preview override; production workflow uses 1080.
        if request.get('preview_width'):
            width = int(request['preview_width'])
            if not 160 <= width <= 1080 or width%2:
                raise ValueError('preview_width must be an even integer between 160 and1080.')
            height = width if request['aspect']=='square' else round(width*16/9/2)*2
        stage('captions')
        write_captions(plan,outdir,width,height)
        stage('render')
        result = render_plan(plan,source,outdir,width,height,int(request['fps']),
                             Path(request['music']).resolve() if request.get('music') else None)
        (outdir/'script.txt').write_text('\n\n'.join(b.get('text','') for b in plan['beats']),encoding='utf-8')
        (outdir/'description.txt').write_text(plan.get('description',request['brief'])+'\n',encoding='utf-8')
        status.update(state='succeeded',stage='complete',elapsed_seconds=round(time.time()-started,2),duration=plan['duration'])
        write_json(outdir/'status.json',status)
        return result
    except Exception as exc:
        status.update(state='failed',error=str(exc),elapsed_seconds=round(time.time()-started,2))
        write_json(outdir/'status.json',status)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Generate a standalone movie recap Short.')
    sub = parser.add_subparsers(dest='command',required=True)
    make = sub.add_parser('generate')
    make.add_argument('--request',required=True,type=Path)
    make.add_argument('--workdir',type=Path,default=Path('work/shorts'))
    make.add_argument('--output',type=Path,default=Path('output/shorts'))
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    args = parser.parse_args(argv)
    try:
        if args.command=='generate':
            result=generate(json.loads(args.request.read_text()),args.workdir,args.output)
            print(json.dumps(result,default=str,indent=2))
        else:
            import importlib.util, shutil, requests
            report = {name:bool(shutil.which(name)) for name in ('ffmpeg','ffprobe')}
            report.update({name:importlib.util.find_spec(name) is not None for name in ('faster_whisper','chatterbox','scenedetect')})
            try:
                response=requests.get(args.ollama_url.rstrip('/')+'/api/tags',timeout=5)
                response.raise_for_status()
                report['ollama_models']=[m['name'] for m in response.json()['models']]
            except requests.RequestException:
                report['ollama_models']=None
            print(json.dumps(report,indent=2))
        return 0
    except (RuntimeError,ValueError,OSError,KeyError) as exc:
        print(f'Shorts failed: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
