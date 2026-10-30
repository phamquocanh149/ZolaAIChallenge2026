"""Model code imported only for a real GPU run."""
import time
from pathlib import Path


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class PhoGPT:
    def __init__(self, path, offline=True, context=4096):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU. CPU is only supported for dry-run checks.")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=offline)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=offline)
        cfg.init_device = "cpu"
        cfg.attn_config["attn_impl"] = "torch"
        self.context = min(context, cfg.max_seq_len, 8192)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            path, config=cfg, trust_remote_code=True, local_files_only=offline,
            torch_dtype=dtype, device_map={"": 0}).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if self.parameter_count >= 5_000_000_000:
            raise ValueError("LLM exceeds competition parameter limit")

    def generate(self, prompt, deadline, max_new_tokens=192):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.perf_counter() >= deadline

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs.input_ids.shape[1] + max_new_tokens > self.context:
            raise ValueError("context_overflow: input+output exceeds configured context; no silent truncation")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([Deadline()]))
        return self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ASR:
    def __init__(self, path, offline=True):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=offline)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path, local_files_only=offline, torch_dtype=torch.float16).to("cuda:0").eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def transcribe(self, path, deadline):
        import librosa
        import soundfile as sf
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.perf_counter() >= deadline

        audio, sr = sf.read(str(Path(path)), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Explicit chunking avoids silently discarding speech after Whisper's 30-second window.
        texts = []
        for start in range(0, len(audio), 30 * 16000):
            if time.perf_counter() >= deadline:
                raise TimeoutError("ASR time budget exceeded")
            features = self.processor(audio[start:start + 30 * 16000], sampling_rate=16000,
                                      return_tensors="pt").input_features.to("cuda:0", dtype=self.torch.float16)
            with self.torch.inference_mode():
                ids = self.model.generate(features, language="vi", task="transcribe", do_sample=False,
                                          num_beams=1, max_new_tokens=256,
                                          stopping_criteria=StoppingCriteriaList([Deadline()]))
            texts.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0])
        return " ".join(texts).strip()
