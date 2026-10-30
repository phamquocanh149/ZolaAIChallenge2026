"""Kaggle baseline CLI. Each inference invocation handles exactly one split."""
import argparse
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path

from core import build_prompt, dump, fingerprint, make_split, parse_action, read_rows, resolve_root, tool_f1

DEFAULT_ROOT = "/kaggle/input/datasets/pqa149/zola-train-public-test/train+public_test"


def configure(work):
    work = Path(work).resolve()
    for key, folder in {"HF_HOME": "cache/hf", "HF_HUB_CACHE": "cache/hf/hub",
                        "HF_MODULES_CACHE": "cache/hf/modules", "TORCH_HOME": "cache/torch",
                        "XDG_CACHE_HOME": "cache", "NUMBA_CACHE_DIR": "cache/numba",
                        "PIP_CACHE_DIR": "cache/pip", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp"}.items():
        path = work / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return work


def prepare(args, root, work):
    rows = read_rows(root / "train/train.jsonl")
    split = make_split(rows, seed=args.seed)
    split["train_sha256"] = fingerprint(root / "train/train.jsonl")
    dump(work / "splits.json", split)
    summary = {}
    for name in ("train", "public_test"):
        samples = read_rows(root / name / f"{name}.jsonl")
        missing = []
        for row in samples:
            audio = [row["audio"]] + [h["audio"] for h in row["history"] if h["role"] == "user"]
            missing.extend(p for p in audio if not (root / name / p).is_file())
        summary[name] = {"samples": len(samples), "with_history": sum(bool(r["history"]) for r in samples),
                         "missing_audio": missing}
    summary.update({"dev": len(split["dev"]), "validation": len(split["validation"])})
    dump(work / "data_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(summary[s]["missing_audio"] for s in ("train", "public_test")):
        raise FileNotFoundError("Missing audio; see data_report.json")


def download(args, work):
    from huggingface_hub import HfApi, snapshot_download
    manifest = {}
    for name, repo in (("llm", "vinai/PhoGPT-4B-Chat"), ("asr", "vinai/PhoWhisper-small")):
        revision = HfApi().model_info(repo).sha
        folder = work / "models" / name
        snapshot_download(repo, revision=revision, local_dir=folder,
                          allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.safetensors", "*.bin"])
        manifest[name] = {"repo": repo, "revision": revision, "path": str(folder)}
    dump(work / "models/manifest.json", manifest)
    print("Models saved; inference uses only these local files.")


def load_runtime(root, split):
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Use separate processes for different splits; zaic_tools must not be reused")
    sys.path.insert(0, str(root / split))
    return importlib.import_module("zaic_tools")


def agent(item, tools, env, model, deadline, max_steps=12):
    selected = ["sys.ask_missing_info"] if "sys.ask_missing_info" in tools else []
    observations = []
    failures = 0
    seen = set()
    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            return observations, "timeout"
        try:
            raw = model.generate(build_prompt(item, tools, selected, observations), deadline)
        except Exception as exc:
            observations.append({"error": f"{type(exc).__name__}: {exc}"})
            return observations, "error"
        if time.perf_counter() >= deadline:
            observations.append({"raw": raw, "error": "generation reached deadline"})
            return observations, "timeout"
        try:
            action = parse_action(raw)
            if action["action"] == "finish":
                return observations, "finish"
            if action["action"] == "inspect":
                names = action["names"]
                if not names or any(n not in tools for n in names):
                    raise ValueError("Inspect only listed tool names")
                if all(n in selected for n in names):
                    raise ValueError("Schema already provided; choose call or finish")
                # Keep full history/results; fail explicitly if model context no longer fits.
                selected = list(dict.fromkeys(selected + names))
                continue
            name, arguments = action["name"], action["arguments"]
            if name not in tools:
                raise ValueError("Tool not allowed for this sample")
            if name not in selected:
                selected.append(name)
                observations.append({"error": f"Review schema of {name}, then emit the call again"})
                continue
            validation = env.validate(name, arguments)
            if not validation["ok"]:
                raise ValueError(validation["error"])
            signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            if signature in seen:
                raise ValueError("Identical call repeated without an intervening write")
            result = env.call(name, arguments)
            if tools[name].get("kind") == "write":
                seen.clear()
            seen.add(signature)
            observations.append({"call": {"name": name, "arguments": arguments}, "result": result})
            failures = 0
            if name == "sys.ask_missing_info":
                return observations, "asked_missing_info"
        except (ValueError, TypeError, KeyError) as exc:
            failures += 1
            observations.append({"raw": raw, "error": str(exc)})
            if failures >= 2:
                return observations, "invalid_output"
    return observations, "step_limit"


def input_only(row, root, split, asr, oracle, deadline):
    # Explicit allowlist: labels and final_state never enter the agent prompt.
    item = {"context": copy.deepcopy(row["context"]), "history": []}
    for turn in row["history"]:
        if turn["role"] == "user":
            text = turn["audio_text"] if oracle else asr.transcribe(root / split / turn["audio"], deadline)
            item["history"].append({"role": "user", "text": text})
        else:
            item["history"].append({k: copy.deepcopy(turn[k]) for k in
                                    ("role", "text", "tool_calls", "tool_results") if k in turn})
    item["request"] = row["audio_text"] if oracle else asr.transcribe(root / split / row["audio"], deadline)
    return item


def infer(args, root, work):
    if args.oracle and args.split != "train":
        raise ValueError("--oracle is only for train validation; forbidden on test")
    if args.split != "train" and args.subset != "all":
        raise ValueError("Test splits require --subset all")
    if args.evaluate and args.split != "train":
        raise ValueError("Test labels are unavailable; --evaluate is train-only")
    rows = read_rows(root / args.split / f"{args.split}.jsonl")
    if args.subset != "all":
        split = json.loads((work / "splits.json").read_text(encoding="utf-8"))
        if split["train_sha256"] != fingerprint(root / "train/train.jsonl"):
            raise ValueError("Dataset changed; regenerate splits")
        allowed = set(split[args.subset])
        rows = [r for r in rows if r["id"] in allowed]
    if args.limit:
        rows = rows[:args.limit]
    # New directory required: never silently mix/overwrite runs or resume mutated sessions.
    out = work / "runs" / args.name
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "config.json", vars(args))
    runtime = load_runtime(root, args.split)
    catalog = json.loads((root / args.split / f"{args.split}_tools_catalog.json").read_text(encoding="utf-8"))
    all_tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    model = asr = None
    if not args.dry_run:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from models import ASR, PhoGPT, seed_all
        seed_all(args.seed)
        model = PhoGPT(args.llm or str(work / "models/llm"), context=args.context)
        if not args.oracle:
            asr = ASR(args.asr or str(work / "models/asr"))
        parameters = model.parameter_count + (asr.parameter_count if asr else 0)
        if parameters >= 6_000_000_000:
            raise ValueError("Total parameters exceed competition limit")
        dump(out / "models.json", {"loaded_parameters": parameters, "context_limit": model.context,
                                    "torch": model.torch.__version__})
    metrics = []
    filename = "submission.DRY_RUN.jsonl" if args.dry_run else "submission.jsonl"
    with (out / filename).open("w", encoding="utf-8") as submission, (out / "traces.jsonl").open("w", encoding="utf-8") as traces:
        for row in rows:
            started = time.perf_counter()
            deadline = started + args.seconds
            env = runtime.ToolEnv(args.split, row["id"], root=root)  # exactly once
            # Evaluator only. Never used by agent; no access to files under *_environment.
            expected = None
            if args.evaluate:
                expected = env.get_state()
                expected.update(copy.deepcopy(row["label"]["final_state"]))
            item, events, reason = None, [], "dry_run"
            try:
                if not args.dry_run:
                    item = input_only(row, root, args.split, asr, args.oracle, deadline)
                    names = {n for app in row["context"]["installed_apps"].values() for n in app["tools"]}
                    names.add("sys.ask_missing_info")
                    tools = {n: t for n, t in all_tools.items() if n in names}
                    events, reason = agent(item, tools, env, model, deadline)
            except Exception as exc:
                reason = "error"
                events.append({"error": f"{type(exc).__name__}: {exc}"})
            elapsed = time.perf_counter() - started
            result = {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
            trace = {"id": row["id"], "input": item, "events": events, "stop_reason": reason,
                     "seconds": elapsed, "over_30_seconds": elapsed > 30,
                     "within_budget": elapsed <= args.seconds}
            if args.evaluate:
                f1 = tool_f1(result["tool_calls"], row["label"]["tool_calls"], all_tools)
                state_ok = runtime.state_hash(expected) == result["state_hash"]
                proxy = (float(not result["tool_calls"]) if not row["label"]["tool_calls"] else f1 * state_ok)
                trace["metrics"] = {"tool_f1_proxy": f1, "state_match": state_ok, "score_proxy": proxy,
                                    "score_proxy_30s": proxy if elapsed <= 30 and reason != "error" else 0}
                metrics.append(trace["metrics"])
            submission.write(json.dumps(result, ensure_ascii=False) + "\n")
            submission.flush()
            traces.write(json.dumps(trace, ensure_ascii=False) + "\n")
            traces.flush()
            print(f"{row['id']} calls={len(result['tool_calls'])} stop={reason} {elapsed:.1f}s", flush=True)
    if metrics:
        dump(out / "metrics.json", {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]})
    check_submission(root, args.split, out / filename, rows)
    print(f"Saved {out / filename}. {'FORMAT ONLY; NOT a model prediction.' if args.dry_run else ''}")


def check_submission(root, split, path, expected=None):
    rows = read_rows(path)
    expected = expected if expected is not None else read_rows(root / split / f"{split}.jsonl")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in expected}:
        raise ValueError("Submission has missing, extra, or duplicate IDs")
    import re
    for row in rows:
        if set(row) != {"id", "tool_calls", "state_hash"} or not re.fullmatch(r"[0-9a-f]{64}", row["state_hash"]):
            raise ValueError("Invalid submission structure/hash")
        if not isinstance(row["tool_calls"], list):
            raise ValueError("tool_calls must be a list")
        for call in row["tool_calls"]:
            if set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                raise ValueError("Invalid tool call shape")
    print(f"Submission IDs/format OK: {len(rows)} rows (not replayed; no extra ToolEnv created)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "download", "infer", "check"])
    p.add_argument("--data-root", default=DEFAULT_ROOT)
    p.add_argument("--work-dir", default="/kaggle/working/pipeline_baseline")
    p.add_argument("--seed", type=int, default=149)
    p.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    p.add_argument("--subset", default="all", choices=["all", "dev", "validation"])
    p.add_argument("--name", default="baseline")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--llm")
    p.add_argument("--asr")
    p.add_argument("--submission")
    args = p.parse_args()
    if args.limit < 0 or args.seconds <= 0 or not 512 <= args.context <= 8192:
        p.error("Invalid limit, seconds, or context (512..8192)")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        p.error("--name must be a single directory name")
    work = configure(args.work_dir)
    if args.command == "download":
        download(args, work)
        return
    root = resolve_root(args.data_root)
    if args.command == "prepare":
        prepare(args, root, work)
    elif args.command == "infer":
        infer(args, root, work)
    else:
        if not args.submission:
            p.error("check requires --submission")
        check_submission(root, args.split, Path(args.submission))


if __name__ == "__main__":
    main()
