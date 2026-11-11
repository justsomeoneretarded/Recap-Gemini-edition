"""Install the verified CPU payload into this job's owned temporary directory."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

VERSION='0.40.14'
SHA256='993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d'
url=f'https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-amd64.tar.zst'
root=Path(os.environ['RUNNER_TEMP'])/'shorts-ollama'
root.mkdir(exist_ok=True)
minimum_disk = 16 if os.environ.get('SCRIPT_MODEL')=='gemma4:12b' else 10
if shutil.disk_usage(root).free < minimum_disk*1024**3:
    raise RuntimeError(f'The configured model chain needs at least {minimum_disk} GiB free disk after dependency installation. Use a larger runner or smaller configured models.')
memory = os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
cgroup=Path('/sys/fs/cgroup/memory.max')
if cgroup.is_file() and cgroup.read_text().strip().isdigit():
    memory=min(memory,int(cgroup.read_text().strip()))
minimum_ram=14 if os.environ.get('SCRIPT_MODEL')=='gemma4:12b' else 10
if memory<minimum_ram*1024**3:
    raise RuntimeError(f'This configured model chain requires at least {minimum_ram} GiB RAM; select a larger runner.')
archive=root/'runtime.tar.zst'
subprocess.run(['curl','--fail','--location','--retry','2','--max-time','600','--output',str(archive),url],check=True)
with archive.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=SHA256:
        raise RuntimeError('Ollama release archive checksum mismatch.')
subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(root),'--no-same-owner',
                '--exclude=lib/ollama/cuda_v*','--exclude=./lib/ollama/cuda_v*',
                '--exclude=lib/ollama/vulkan','--exclude=./lib/ollama/vulkan'],check=True)
archive.unlink()
binary=root/'bin'/'ollama'
if not binary.is_file():
    raise RuntimeError('Ollama archive has no executable.')
subprocess.run([str(binary),'--version'],check=True)
