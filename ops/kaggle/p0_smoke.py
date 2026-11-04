"""P0, bước chạy 1 trên Kaggle (T4): cài vLLM, kiểm tra `import vllm` và các giả định API mà P1 cần.

Chạy như một Kaggle kernel dạng script (ops/kaggle/push.py đẩy lên). Không dùng tập kiểm tra.
Kết quả ghi vào /kaggle/working/p0_smoke.json để tải về bằng `kaggle kernels output`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

VLLM_VERSION = os.environ.get("TTALLOC_VLLM", "0.32.0")
MODEL = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
RUN_ID = globals().get("TTALLOC_RUN_ID")  # push.py chèn vào; trình theo dõi dùng để bỏ qua output của bản trước
REPORT: dict[str, Any] = {"run_id": RUN_ID, "vllm_version_requested": VLLM_VERSION, "steps": {}}


def save() -> None:
    (WORK / "p0_smoke.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")


def step(name: str):  # type: ignore[no-untyped-def]
    def deco(fn):  # type: ignore[no-untyped-def]
        def run(*args: Any, **kwargs: Any) -> Any:
            t0 = time.time()
            print(f"=== {name}", flush=True)
            try:
                out = fn(*args, **kwargs)
                REPORT["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": out}
                return out
            except Exception as exc:  # ghi lại mọi lỗi để đọc sau, rồi chạy tiếp bước khác
                REPORT["steps"][name] = {
                    "ok": False,
                    "seconds": round(time.time() - t0, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
                print(REPORT["steps"][name]["traceback"], flush=True)
                return None
            finally:
                save()

        return run

    return deco


def sh(cmd: list[str], timeout: int = 3600) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    tail = (out.stdout + out.stderr)[-3000:]
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> mã {out.returncode}\n{tail}")
    return tail


@step("gpu_info")
def gpu_info() -> dict[str, Any]:
    smi = sh(["nvidia-smi"])
    m = re.search(r"Driver Version:\s*([\d.]+).*?CUDA Version:\s*([\d.]+)", smi, re.S)
    q = sh(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"])
    return {"driver": m.group(1) if m else None, "cuda_max": m.group(2) if m else None, "gpus": q.strip().splitlines()}


@step("find_source")
def find_source() -> str:
    """Tìm mã nguồn ttalloc: dataset Kaggle đính kèm (thư mục hoặc tệp ttalloc-src.tar.gz)."""
    root = Path("/kaggle/input")
    target = Path(tempfile.gettempdir()) / "ttalloc"  # ngoài /kaggle/working để cây mã không thành output
    for tgz in root.rglob("ttalloc-src.tar.gz"):
        if target.exists():
            shutil.rmtree(target)
        with tarfile.open(tgz) as tf:
            tf.extractall(target, filter="data")
        return str(target)
    for pp in root.rglob("pyproject.toml"):
        if 'name = "ttalloc"' in pp.read_text(encoding="utf-8"):
            shutil.copytree(pp.parent, target, dirs_exist_ok=True)
            return str(target)
    raise FileNotFoundError("không thấy mã nguồn ttalloc trong /kaggle/input (đính kèm dataset ttalloc-src)")


@step("pip_install")
def pip_install(src: str | None) -> dict[str, str]:
    logs = {"vllm": sh([sys.executable, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"])}
    logs["deps"] = sh([sys.executable, "-m", "pip", "install", "-q", "math-verify>=0.7", "pydantic>=2.8", "typer"])
    if src:
        # Kaggle dùng Python 3.13, pyproject khai báo <3.13 (máy Vast và máy cá nhân dùng 3.12).
        logs["ttalloc"] = sh(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python", "-e", src]
        )
    return {k: v[-500:] for k, v in logs.items()}


@step("import_vllm")
def import_vllm() -> dict[str, Any]:
    import torch
    import vllm

    return {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


@step("engine_checks")
def engine_checks() -> dict[str, Any]:
    """Các giả định của PLAN mục 3.3 và A1: TokensPrompt, logprob token đã sinh, chuỗi dừng, prefix caching."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from ttalloc.config import load_model_cfg
    from ttalloc.data.prompts import build_prompt_ids, load_prompts
    from ttalloc.types import Question

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    mcfg = load_model_cfg("qwen3_1_7b")
    q = Question("smoke/0", "smoke", "What is 7 * 8 + 3?", None, "59", "math", None)
    prompt_ids = build_prompt_ids(q, mcfg, tok, load_prompts())
    llm = LLM(
        model=MODEL,
        revision=MODEL_REVISION,
        dtype="float16",
        max_model_len=8192,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    res: dict[str, Any] = {"prompt_len": len(prompt_ids)}
    sp = SamplingParams(
        max_tokens=512,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        logprobs=0,
        seed=0,
        stop=["\n\nWait", "\n\nBut", "\n\nHmm"],
        include_stop_str_in_output=True,
    )
    t0 = time.time()
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], [sp], use_tqdm=False)[0].outputs[0]
    res["segment"] = {
        "n_tokens": len(out.token_ids),
        "finish_reason": out.finish_reason,
        "stop_reason": out.stop_reason,
        "n_logprobs": len(out.logprobs or []),
        "first_logprob_keys": [list(d.keys())[:2] for d in (out.logprobs or [])[:2]],
        "text_tail": out.text[-200:],
        "seconds": round(time.time() - t0, 2),
    }
    chain = list(prompt_ids) + list(out.token_ids)
    probe_ids = tok(load_prompts().probe_math, add_special_tokens=False)["input_ids"]
    sp_probe = SamplingParams(max_tokens=20, temperature=0.0, logprobs=0, stop=["}"], include_stop_str_in_output=True)
    t0 = time.time()
    pout = llm.generate([TokensPrompt(prompt_token_ids=chain + probe_ids)], [sp_probe], use_tqdm=False)[0].outputs[0]
    res["probe"] = {"text": pout.text, "n_tokens": len(pout.token_ids), "seconds": round(time.time() - t0, 2)}
    t0 = time.time()
    llm.generate([TokensPrompt(prompt_token_ids=chain + probe_ids)], [sp_probe], use_tqdm=False)
    res["probe_repeat_seconds"] = round(time.time() - t0, 2)  # nhanh hơn rõ rệt nếu prefix caching hoạt động
    think_end = tok.convert_tokens_to_ids("</think>")
    sp_long = SamplingParams(max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, seed=1, stop_token_ids=[think_end])
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * 32, [sp_long] * 32, use_tqdm=False)
    n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
    dt = time.time() - t0
    res["throughput_batch32"] = {"tokens": n_gen, "seconds": round(dt, 1), "tok_per_s": round(n_gen / dt, 1)}
    res["think_end_id"] = think_end
    res["stop_reasons"] = sorted({str(o.outputs[0].stop_reason) for o in outs})
    return res


def main() -> None:
    print(f"RUN_ID {RUN_ID}", flush=True)
    gpu_info()
    src = find_source()
    pip_install(src)
    if src and src not in sys.path:
        sys.path.insert(0, str(Path(src) / "src"))
    if import_vllm() is not None:
        engine_checks()
    REPORT["ok"] = all(s.get("ok") for s in REPORT["steps"].values())
    save()
    print(json.dumps({k: v.get("ok") for k, v in REPORT["steps"].items()}, indent=2))


if __name__ == "__main__":
    main()
