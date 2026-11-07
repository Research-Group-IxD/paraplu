#!/usr/bin/env python3
"""
Six Villages: Mac health check ("doctor")

Run this when the POC is slow or the Mac starts lagging:
  python3 doctor.py

It looks at memory, leftover POC processes and what Ollama has loaded, then
asks the model two small questions and times them. Everything is printed and
also saved to logs/doctor_<date>_<time>.txt, so it can be read afterwards.
Nothing is changed or deleted. Takes about 1 to 2 minutes.
"""

import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import poc   # reuse the real village prompt and the same settings as the POC

OLLAMA = "http://localhost:11434"
OUT = poc.HERE / "logs" / f"doctor_{datetime.now():%Y-%m-%d_%H%M}.txt"
lines = []


def say(text=""):
    """Print and remember, so the same text ends up in the report file."""
    print(text)
    lines.append(text)


def sh(cmd):
    """Runs a small Mac command and returns its output (or why it failed)."""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e:
        return f"(could not run: {e})"


def get(path, timeout=10):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.load(r)


def post(path, payload, timeout=180):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gb(n):
    return f"{(n or 0) / 1e9:.1f} GB"


def loaded_models():
    """What Ollama holds in memory right now, and how much of it sits on the GPU."""
    try:
        models = get("/api/ps").get("models", [])
    except Exception as e:
        say(f"  could not ask Ollama: {e}")
        return
    if not models:
        say("  nothing loaded")
    for m in models:
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        gpu = round(100 * vram / size) if size else 0
        say(f"  {m.get('name')}: {gb(size)} in memory, {gpu}% on the GPU, "
            f"context {m.get('context_length', '?')} tokens, unloads at {m.get('expires_at', '?')[:19]}")
        if gpu < 100:
            say("  !! part of the model runs on the CPU: that is the slow, laggy case")


def timed_chat(label, messages, num_predict):
    """One real request with the POC's own settings; Ollama reports where the time went."""
    payload = {"model": poc.DEFAULT_MODEL, "messages": messages, "stream": False,
               "options": dict(poc.MODEL_OPTIONS, num_predict=num_predict),
               "keep_alive": "5m"}
    start = time.perf_counter()
    try:
        r = post("/api/chat", payload)
    except Exception as e:
        say(f"  {label}: FAILED after {time.perf_counter() - start:.0f} s ({e})")
        return
    ns = 1e9
    load = r.get("load_duration", 0) / ns
    p_n, p_t = r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 1) / ns
    e_n, e_t = r.get("eval_count", 0), r.get("eval_duration", 1) / ns
    say(f"  {label}: {time.perf_counter() - start:.1f} s total | loading model {load:.1f} s | "
        f"reading prompt {p_n} tokens in {p_t:.1f} s | writing {e_n} tokens at {e_n / e_t:.1f} tokens/s")
    say(f"  answer: {r.get('message', {}).get('content', '').strip()[:160]}")


def main():
    say(f"Six Villages doctor, {datetime.now():%Y-%m-%d %H:%M}")
    say(f"POC settings: model {poc.DEFAULT_MODEL}, options {poc.MODEL_OPTIONS}, keep_alive {poc.KEEP_ALIVE}")

    say("\n1. Memory")
    say(f"  RAM: {gb(int(sh('sysctl -n hw.memsize') or 0))}")
    say("  " + (sh("memory_pressure -Q | tail -1") or "(memory_pressure not available)"))
    say("  swap: " + sh("sysctl -n vm.swapusage"))
    say("  biggest programs (MB):")
    for row in sh("ps -axo rss=,comm= | sort -rn | head -8").splitlines():
        rss, _, name = row.strip().partition(" ")
        say(f"    {int(rss) // 1024:>6}  {name.strip().split('/')[-1]}")

    say("\n2. Leftovers from earlier runs")
    say("  POC processes: " + (sh("ps -axo pid=,command= | grep -E 'poc[.]py|demo_page[.]py'") or "none"))
    say("  port 8000: " + (sh("lsof -nP -iTCP:8000 -sTCP:LISTEN | tail -n +2") or "free"))

    say("\n3. Ollama")
    try:
        say(f"  version {get('/api/version').get('version')}")
        for m in get("/api/tags").get("models", []):
            say(f"  installed: {m['name']} ({gb(m.get('size'))})")
    except Exception as e:
        say(f"  Ollama does not answer ({e}). Open the Ollama app and run this again.")
        return finish()
    say("  loaded before the test:")
    loaded_models()

    say("\n4. Speed test (same settings as the POC)")
    timed_chat("tiny question", [{"role": "user", "content": "Say OK."}], 5)
    gestel = poc.load_village("gestel")
    timed_chat("real Gestel question", [
        {"role": "system", "content": poc.system_prompt(gestel, [], "en")},
        {"role": "user", "content": "Who painted you, and how often?\n\n" + poc.LANGUAGE_REMINDERS["en"]}],
        poc.MODEL_OPTIONS["num_predict"])
    say("  loaded after the test:")
    loaded_models()
    finish()


def finish():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
