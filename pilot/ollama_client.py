"""Minimal client for a local ollama server (standard library only)."""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:11434"
_server = None


def is_up() -> bool:
    try:
        urllib.request.urlopen(HOST + "/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def start_server(parallel: int = 4, models_dir: str = "/tmp/ollama") -> None:
    """Install ollama if needed and start the server in the background."""
    global _server
    if is_up():
        return
    if shutil.which("ollama") is None:
        if shutil.which("zstd") is None:   # the ollama installer needs zstd to unpack
            subprocess.run("apt-get install -y -qq zstd", shell=True, check=True)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(parallel), OLLAMA_MAX_LOADED_MODELS="2",
               OLLAMA_KEEP_ALIVE="60m", OLLAMA_MODELS=models_dir)
    log = open("/tmp/ollama_server.log", "a")
    _server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log)
    for _ in range(60):
        if is_up():
            return
        time.sleep(1)
    raise RuntimeError("ollama server did not start; see /tmp/ollama_server.log")


def stop_server() -> None:
    """Stop the server so the GPUs are released for training."""
    global _server
    if _server is not None:
        _server.terminate()
        _server.wait(timeout=60)
        _server = None
    subprocess.run(["pkill", "-f", "ollama"], check=False)
    time.sleep(5)


def pull(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def chat(model: str, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0,
         seed: int = 0, max_tokens: int = 300, retries: int = 3, fallback: str | None = None) -> str:
    """One chat reply. If every try fails, return `fallback` when one is given, otherwise raise."""
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens,
                           "num_ctx": 4096}}
    if json_mode:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(HOST + "/api/chat", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read())["message"]["content"].strip()
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(5 * (attempt + 1))
    if fallback is not None:
        print(f"  ollama chat gave up on one request: {last_error}", flush=True)
        return fallback
    raise RuntimeError(f"ollama chat failed after {retries} tries: {last_error}")


def chat_many(model: str, batch: list[list[dict]], workers: int = 4, **kwargs) -> list[str]:
    """Run many chats in parallel, keeping the input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda messages: chat(model, messages, **kwargs), batch))
