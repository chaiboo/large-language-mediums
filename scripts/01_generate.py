"""Generate model responses for the pushback 2x2 items.

Runs every prompt in data/items.json (24 items x 5 framings = 120 prompts)
through the test-set models via OpenRouter, N_DRAWS times each, and writes
data/responses.parquet. Resumable: reruns skip (item, cell, model, draw) rows
already on disk.

Design decisions carried from the wildchat-rel-ref audit:
- N_DRAWS = 5 draws per cell at TEMPERATURE = 0.7, so confidence intervals can
  include sampling variance. One draw per cell was the old design's flaw.
- The judge checkpoints (openai/gpt-4o, google/gemini-2.5-flash) are NOT in the
  test set. Judge-family subjects that remain (GPT-4o-mini, Gemini-3.5-Flash)
  get per-family agreement reporting at the judge stage.
- Every row records model id, display name, lab, draw index, temperature, and
  timestamp, so no number downstream lacks provenance.

Cost estimate before running: 120 prompts x 8 models x 5 draws = 4,800
completions at max 600 tokens.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data" / "items.json"
OUT = ROOT / "data" / "responses.parquet"

N_DRAWS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 600
CELLS = ["secular", "testable_unprotected", "testable_protected",
         "untestable_unprotected", "untestable_protected"]

# Test set. Judges (openai/gpt-4o, google/gemini-2.5-flash) intentionally absent.
MODELS = [
    ("openai/gpt-4o-mini", "GPT-4o-mini", "OpenAI"),
    ("anthropic/claude-sonnet-4.6", "Claude-Sonnet-4.6", "Anthropic"),
    ("google/gemini-3.5-flash", "Gemini-3.5-Flash", "Google"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B", "Meta"),
    ("qwen/qwen-2.5-72b-instruct", "Qwen2.5-72B", "Alibaba"),
    ("deepseek/deepseek-chat", "DeepSeek-Chat", "DeepSeek"),
    ("cohere/command-r-plus-08-2024", "Command-R+", "Cohere"),
    ("mistralai/mistral-nemo", "Mistral-Nemo", "Mistral"),
]


def load_key() -> str:
    for env_path in (ROOT / ".env", ROOT.parent / "wildchat-rel-ref" / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not found in .env here or in ../wildchat-rel-ref/.env")


def call(key: str, model_id: str, prompt: str, retries: int = 3) -> str:
    for i in range(retries):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model_id,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
                timeout=90,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code in (429, 502, 503):
                time.sleep(3 * (i + 1))
                continue
            return f"__ERROR_{r.status_code}__ {r.text[:120]}"
        except Exception as e:
            time.sleep(2 * (i + 1))
            if i == retries - 1:
                return f"__EXC__ {type(e).__name__}"
    return "__ERROR_retries__"


def main() -> None:
    key = load_key()
    items = json.loads(ITEMS.read_text(encoding="utf-8"))

    rows: list[dict] = []
    done: set[tuple] = set()
    if OUT.exists():
        d = pd.read_parquet(OUT)
        rows = d.to_dict("records")
        done = set(zip(d["item"], d["cell"], d["model"], d["draw"]))
        print(f"Resuming: {len(done)} responses already on disk")

    jobs = []
    for it in items:
        for cell in CELLS:
            prompt = it["prompts"][cell]
            for model_id, disp, lab in MODELS:
                for draw in range(N_DRAWS):
                    if (it["item"], cell, disp, draw) in done:
                        continue
                    jobs.append((it, cell, prompt, model_id, disp, lab, draw))

    print(f"{len(jobs)} completions to fetch "
          f"({len(items)} items x {len(CELLS)} cells x {len(MODELS)} models x {N_DRAWS} draws)")

    def run(job):
        it, cell, prompt, model_id, disp, lab, draw = job
        text = call(key, model_id, prompt)
        return {
            "item": it["item"], "stakes": it["stakes"], "domain": it["domain"],
            "cell": cell,
            "practice": it["practices"].get(cell, "none"),
            "outcome_claim": it["outcome_claim"],
            "prompt": prompt,
            "model": disp, "model_id": model_id, "lab": lab,
            "draw": draw, "temperature": TEMPERATURE,
            "response": text,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    n_new = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            n_new += 1
            if n_new % 50 == 0:
                pd.DataFrame(rows).to_parquet(OUT, index=False)
                print(f"  {n_new}/{len(jobs)} fetched, checkpointed")

    pd.DataFrame(rows).to_parquet(OUT, index=False)
    errs = sum(1 for r in rows if str(r["response"]).startswith(("__ERROR", "__EXC")))
    print(f"Done: {len(rows)} rows total, {errs} errors. Output: {OUT}")


if __name__ == "__main__":
    main()
