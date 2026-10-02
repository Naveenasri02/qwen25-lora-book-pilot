"""Merge the adapter into the base, convert to q8 GGUF, and register it with ollama.

The model is examined in the form it would be deployed in: merged, q8, served by ollama.

Merging runs as its own process so the memory is released afterwards:
    python -m pilot.export merge --adapter adapter_dir --out merged_dir
"""
import argparse
import subprocess
import sys
from pathlib import Path

from .config import BASE_HF, BASE_OLLAMA

LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"


def merge(adapter_dir: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()   # the per-weight progress lines flood the notebook log
    base =AutoModelForCausalLM.from_pretrained(BASE_HF, torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(BASE_HF).save_pretrained(merged_dir)
    print(f"merged model saved to {merged_dir}")


def to_gguf_q8(merged_dir: str, gguf_path: str, llama_dir: str = "/tmp/llama.cpp") -> None:
    if not Path(llama_dir).exists():
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP, llama_dir], check=True)
    subprocess.run([sys.executable, f"{llama_dir}/convert_hf_to_gguf.py", merged_dir,
                    "--outfile", gguf_path, "--outtype", "q8_0"], check=True)


def ollama_create(gguf_path: str, name: str, workdir: str = "/tmp") -> None:
    """Register the GGUF with ollama, reusing the base model's chat template and parameters."""
    shown = subprocess.run(["ollama", "show", "--modelfile", BASE_OLLAMA], check=True,
                           capture_output=True, text=True).stdout
    lines = [ln for ln in shown.splitlines() if not ln.startswith("FROM ") and not ln.startswith("#")]
    modelfile = Path(workdir) / "Modelfile"
    modelfile.write_text(f"FROM {gguf_path}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["merge"])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        merge(a.adapter, a.out)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)   # stdout, so the reason is kept in the notebook log
        sys.exit(1)
