"""Portable Clef launcher using pinned HF CLI downloads and observed server metadata."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import psutil
from .core import ROOT,load_json,write_json,sha256


def serve(args):
    weights=load_json(args.weights)
    executable=args.llama_server or os.environ.get('LLAMA_SERVER') or shutil.which('llama-server')
    if executable:executable=shutil.which(executable) or executable
    if not executable and Path('/usr/local/lib/ollama/llama-server').exists():executable='/usr/local/lib/ollama/llama-server'
    if not executable or not Path(executable).is_file():raise ValueError('Set --llama-server or LLAMA_SERVER to a Clef-capable executable')
    if args.context<1 or args.threads<1 or not 1<=args.port<=65535:raise ValueError('Invalid server context, thread count or port')
    command=[str(Path(executable).resolve()),'--model','<model.gguf>','--mmproj','<projector.gguf>','--host',args.host,'--port',str(args.port),
             '-c',str(args.context),'-np','1','-t',str(args.threads),'-tb',str(args.threads),'-ngl',str(args.gpu_layers)]
    if args.dry_run:return dict(command=command,weights=weights,note='No weights downloaded and no server started. Requires Clef/System One support.')
    hf=shutil.which('hf')
    adjacent=Path(sys.executable).parent/('hf.exe' if os.name=='nt' else 'hf')
    if adjacent.exists():hf=str(adjacent)
    if not hf:raise ValueError('Install requirements/models.txt to provide the hf CLI')
    files={}
    for info in weights['files']:
        proc=subprocess.run([hf,'download',weights['repository'],info['filename'],'--revision',weights['revision'],'--quiet'],capture_output=True,text=True,check=True)
        path=Path(proc.stdout.strip().splitlines()[-1]);observed=sha256(path)
        if observed!=info['sha256']:raise ValueError(f'Weight checksum mismatch: {info["filename"]}')
        files[info['filename']]=dict(path=str(path),sha256=observed,size=path.stat().st_size)
    model=next(v['path'] for k,v in files.items() if not k.startswith('mmproj'))
    projector=next(v['path'] for k,v in files.items() if k.startswith('mmproj'))
    command[2]=model;command[4]=projector
    version=subprocess.run([str(executable),'--version'],capture_output=True,text=True,timeout=15)
    metadata_path=Path(args.metadata).resolve();metadata_path.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(command)
    try:
        write_json(metadata_path,dict(pid=process.pid,process_created_s=psutil.Process(process.pid).create_time(),revision=weights['revision'],
            quantization='Q4_K_M',weights=files,command=command,runtime_version=(version.stdout+version.stderr).strip(),
            endpoint=f'http://{args.host}:{args.port}/v1/systemone'))
        print(f'Server PID: {process.pid}. In the benchmark shell, set CLEF_SERVER_METADATA={metadata_path}',flush=True)
        code=process.wait()
        if code:raise RuntimeError(f'llama-server exited with code {code}; check Clef support and memory')
    except KeyboardInterrupt:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    return dict(status='server-stopped',metadata=str(metadata_path))
