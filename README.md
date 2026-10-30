# ZolaAIChallenge2026

Baseline LifeMate: PhoWhisper-small → PhoGPT-4B-Chat → tool execution → submission.

Source, standalone Kaggle notebook and detailed instructions:
[pipeline_baseline/README.md](pipeline_baseline/README.md).

## Run from a Kaggle notebook

Enable GPU and Internet for installation/model preparation, then clone:

```python
!git clone https://github.com/phamquocanh149/ZolaAIChallenge2026.git /kaggle/working/zola-repo
%cd /kaggle/working/zola-repo/pipeline_baseline
```

Install and configure the runner:

```python
import os, sys, subprocess
from pathlib import Path

DATA = '/kaggle/input/datasets/pqa149/zola-train-public-test/train+public_test'
WORK = '/kaggle/working/pipeline_baseline'
SCRIPT = str(Path('run.py').resolve())
for key, folder in {'PIP_CACHE_DIR': 'cache/pip', 'TMPDIR': 'tmp'}.items():
    path = Path(WORK) / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
subprocess.run([sys.executable, 'setup_kaggle.py', '--work-dir', WORK], check=True)
PYTHON = f'{WORK}/.venv311/bin/python'

def run(*args):
    subprocess.run([PYTHON, SCRIPT, *args,
                    '--data-root', DATA, '--work-dir', WORK], check=True)

run('prepare')
run('download')
```

Smoke test before running the full public split:

```python
run('infer', '--split', 'train', '--subset', 'dev', '--oracle', '--evaluate',
    '--limit', '5', '--seconds', '120', '--name', 'smoke_oracle_v1')
run('infer', '--split', 'train', '--subset', 'dev', '--evaluate',
    '--limit', '5', '--name', 'smoke_audio_v1')
```

Full public inference and submission format check:

```python
run('infer', '--split', 'public_test', '--name', 'public_v1')
submission = f'{WORK}/runs/public_v1/submission.jsonl'
run('check', '--split', 'public_test', '--submission', submission)
import shutil
shutil.copyfile(submission, '/kaggle/working/submission.jsonl')
```

Use a new `--name` for each experiment. Inference uses local models offline.
The 30-second sample deadline is soft; GPU quality, memory and latency must be measured.
Local metrics are proxies, not the official scorer. No GPU accuracy results are claimed.

Only code is included. Competition data, model weights, caches and prediction artifacts
are excluded. Attach the competition dataset separately on Kaggle; do not redistribute it here.
