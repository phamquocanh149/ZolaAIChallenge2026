"""Bootstrap an isolated Python 3.11 runtime; launch from the Kaggle kernel Python."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='/kaggle/working/pipeline_baseline')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('This installer targets Kaggle Linux, not Windows.')
    work = Path(args.work_dir).resolve()
    env = os.environ.copy()
    for key, suffix in {'PIP_CACHE_DIR': 'cache/pip', 'UV_CACHE_DIR': 'cache/uv',
                        'UV_PYTHON_INSTALL_DIR': 'python', 'UV_PYTHON_BIN_DIR': 'bin',
                        'TMPDIR': 'tmp'}.items():
        directory = work / suffix
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    # Do not inherit user/global pip installation destinations or Python module paths.
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PIP_CONFIG_FILE'] = os.devnull

    def run(*cmd):
        subprocess.run(list(map(str, cmd)), env=env, check=True)

    # Install only the bootstrap utility into a project-local directory.
    bootstrap = work / 'bootstrap'
    run(sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
        '--target', bootstrap, 'uv==0.9.7')
    uv = bootstrap / 'bin/uv'
    if not uv.is_file():
        raise FileNotFoundError(f'uv executable not found: {uv}')
    venv = work / '.venv311'
    python = venv / 'bin/python'
    if not python.is_file():
        run(uv, 'venv', '--python', '3.11', '--seed', venv)
    run(python, '-c', 'import sys; assert sys.version_info[:2] == (3,11), sys.version')
    run(python, '-m', 'pip', 'install', '--only-binary=:all:', 'torch==2.5.1',
        '--index-url', 'https://download.pytorch.org/whl/cu121')
    run(python, '-m', 'pip', 'install', '--only-binary=:all:', '-r',
        Path(__file__).resolve().with_name('requirements.txt'))
    run(python, '-m', 'pip', 'check')
    run(python, '-c', '''import sys, torch, transformers, tokenizers, sentencepiece, librosa, numpy
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Transformers:', transformers.__version__, 'NumPy:', numpy.__version__)
assert torch.cuda.is_available(), 'Enable GPU in Kaggle settings before running the pipeline'
print('GPU:', torch.cuda.get_device_name(0))
''')
    print(f'Ready. Use this executable for ALL pipeline commands: {python}')


if __name__ == '__main__':
    main()
