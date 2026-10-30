"""Build a standalone Kaggle notebook with embedded source, no dataset bundled."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": s.splitlines(keepends=True),
                  "outputs": [], "execution_count": None})


md("""# LifeMate — PhoGPT baseline trên Kaggle

GPU + Internet cho bước cài đặt/tải model. Inference dùng local model offline.
Dataset mặc định theo đường dẫn đã cung cấp. Không sửa hay đọc trực tiếp environment files.
Chạy theo thứ tự. Cell public chỉ chạy khi bật `RUN_PUBLIC=True`.
Đây là zero-shot baseline, chưa fine-tune; chưa được kiểm chứng chất lượng/tốc độ trên GPU.
""")
sources = {n: (ROOT / n).read_text(encoding="utf-8") for n in
           ("core.py", "models.py", "run.py", "requirements.txt", "test_core.py", "README.md", "setup_kaggle.py")}
code('''import os, sys, json, subprocess, pathlib
WORK = pathlib.Path("/kaggle/working/pipeline_baseline")
SRC = WORK / "src"
SRC.mkdir(parents=True, exist_ok=True)
DATA = "/kaggle/input/datasets/pqa149/zola-train-public-test/train+public_test"
for key, suffix in {"PIP_CACHE_DIR":"cache/pip", "HF_HOME":"cache/hf", "TMPDIR":"tmp"}.items():
    directory = WORK / suffix
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)
''' + "SOURCES = " + repr(sources) + '''
for name, content in SOURCES.items():
    (SRC / name).write_text(content, encoding="utf-8")
def run(*args):
    subprocess.run([PYTHON, str(SRC / "run.py"), *map(str, args),
                    "--data-root", DATA, "--work-dir", str(WORK)], check=True)
print("Source:", SRC)
''')
md("Tạo Python 3.11 riêng và cài PyTorch CUDA cùng dependencies. Kernel Kaggle không bị thay đổi.")
code('''subprocess.run([sys.executable, str(SRC / "setup_kaggle.py"), "--work-dir", str(WORK)], check=True)
PYTHON = str(WORK / ".venv311/bin/python")
subprocess.run([PYTHON, "-m", "unittest", "-v", "test_core"], cwd=SRC, check=True)
subprocess.run(["nvidia-smi"], check=True)
run("prepare")
''')
md("""## Chuẩn bị model
Lần đầu bật Internet và tải weights. Để dùng model đã attach từ Kaggle Input,
đặt `DOWNLOAD_MODELS=False` và sửa `LLM_PATH`, `ASR_PATH`.
Quá trình download có thể tốn nhiều GB; lưu version output để tái sử dụng.
""")
code('''DOWNLOAD_MODELS = True
LLM_PATH = str(WORK / "models/llm")
ASR_PATH = str(WORK / "models/asr")
if DOWNLOAD_MODELS:
    run("download")
MODEL_ARGS = ["--llm", LLM_PATH, "--asr", ASR_PATH]
''')
md("""## Smoke test PhoGPT với transcript thật (5 mẫu development)
120 giây/mẫu để chẩn đoán JSON/tool use, không phải kiểm tra giới hạn thi.
Mỗi tên run phải mới. Khi chạy lại notebook, đổi `TAG`.
""")
code('''TAG = "v1"
run("infer", "--split", "train", "--subset", "dev", "--oracle", "--evaluate",
    "--limit", "5", "--seconds", "120", "--name", f"smoke_oracle_{TAG}", *MODEL_ARGS)
print((WORK / "runs" / f"smoke_oracle_{TAG}" / "metrics.json").read_text())
''')
md("""## Smoke test end-to-end (audio → ASR → PhoGPT → tools)
5 mẫu development, deadline 30 giây. Kiểm tra traces trước khi chạy cả validation/public.
""")
code('''run("infer", "--split", "train", "--subset", "dev", "--evaluate", "--limit", "5",
    "--name", f"smoke_audio_{TAG}", *MODEL_ARGS)
print((WORK / "runs" / f"smoke_audio_{TAG}" / "metrics.json").read_text())
''')
md("## Validation cố định — bật cờ để chạy hai chế độ. Không phải điểm leaderboard chính thức.")
code('''RUN_VALIDATION = False
if RUN_VALIDATION:
    run("infer", "--split", "train", "--subset", "validation", "--oracle", "--evaluate",
        "--name", f"val_oracle_{TAG}", *MODEL_ARGS)
    run("infer", "--split", "train", "--subset", "validation", "--evaluate",
        "--name", f"val_audio_{TAG}", *MODEL_ARGS)
''')
md("""## Public test → submission thật
Bật `RUN_PUBLIC=True` để chạy mọi mẫu public. Không có oracle/labels, không tune trên public.
634 mẫu × 30 giây tương đương khoảng 5,3 giờ, chưa tính model loading và vượt deadline mềm.
Không dùng `--limit` cho bài nộp đầy đủ. `submission.jsonl` chỉ được copy sau khi chạy và kiểm tra xong.
""")
code('''RUN_PUBLIC = False
if RUN_PUBLIC:
    import shutil
    run("infer", "--split", "public_test", "--name", f"public_baseline_{TAG}", *MODEL_ARGS)
    submission = WORK / "runs" / f"public_baseline_{TAG}" / "submission.jsonl"
    run("check", "--split", "public_test", "--submission", submission)
    shutil.copyfile(submission, "/kaggle/working/submission.jsonl")
    print("Download /kaggle/working/submission.jsonl")
''')
md("## Tổng hợp stop reason và độ trễ của các run đã hoàn tất hoặc đang có trace")
code('''from collections import Counter
for path in sorted((WORK / "runs").glob("*/traces.jsonl")):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows:
        latencies = sorted(r["seconds"] for r in rows)
        print(path.parent.name, "n=", len(rows),
              "stop=", dict(Counter(r["stop_reason"] for r in rows)),
              "mean_s=", round(sum(latencies)/len(rows), 2),
              "p95_s=", round(latencies[int((len(rows)-1)*.95)], 2),
              "over_30s=", sum(r["over_30_seconds"] for r in rows))
''')
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"baseline-{i:02d}"
(ROOT / "Kaggle_PhoGPT_Baseline.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Built", ROOT / "Kaggle_PhoGPT_Baseline.ipynb")
