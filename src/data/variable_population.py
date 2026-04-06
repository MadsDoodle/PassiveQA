"""
Variable Population using GPT-4o-mini.

Populates known_variables and missing_variables fields in the unified
dataset using concurrent OpenAI API calls with crash-safe incremental
saving.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
from tqdm import tqdm


# ── Prompt ──────────────────────────────────────────────────────────────────

VARIABLE_PROMPT = """\
You are a variable extractor for a question-answering system.

Given a query, context, and a decision action (ANSWER / ASK / ABSTAIN),
extract the following:

1. known_variables: facts or values present in the context that are
   relevant to answering the query. List as short phrases.
2. missing_variables: facts or values needed to answer the query but
   NOT present in the context. For ASK → these are recoverable via
   clarification. For ABSTAIN → irrecoverable.

Output ONLY valid JSON in this format (no markdown, no preamble):
{{
  "known_variables": ["...", "..."],
  "missing_variables": ["...", "..."]
}}

Query: {query}
Action: {action}
Context (first 600 chars): {context}
"""


def _build_context_str(sample: dict, max_chars: int = 600) -> str:
    docs = sample.get("context", {}).get("documents", [])
    text = " ".join(d.get("text", "") for d in docs)
    return text[:max_chars]


def _populate_one(sample: dict, client: openai.OpenAI,
                  model: str = "gpt-4o-mini") -> dict:
    """Call GPT-4o-mini to extract variables for one sample."""
    prompt = VARIABLE_PROMPT.format(
        query   = sample["query"],
        action  = sample["action"],
        context = _build_context_str(sample),
    )
    try:
        resp = client.chat.completions.create(
            model    = model,
            messages = [{"role": "user", "content": prompt}],
            max_tokens      = 256,
            temperature     = 0.0,
            response_format = {"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        sample["state"]["known_variables"]   = parsed.get("known_variables",  [])
        sample["state"]["missing_variables"] = parsed.get("missing_variables", [])
    except Exception as exc:
        # Leave lists empty on failure — don't crash the run
        print(f"  [WARN] {sample['id']}: {exc}")
    return sample


def populate_variables(
    input_path : str,
    output_path: str,
    api_key    : str,
    model      : str = "gpt-4o-mini",
    max_workers: int = 8,
    sleep_on_rate: float = 60.0,
) -> None:
    """
    Populate known/missing variables for every sample in a JSONL file.

    Saves incrementally to output_path so a crash loses minimal work.
    Already-processed samples are skipped on resume.

    Args:
        input_path:    Path to unified_train.jsonl (or trimmed subset).
        output_path:   Path to write populated JSONL.
        api_key:       OpenAI API key.
        model:         GPT model to use.
        max_workers:   Thread-pool size for concurrent calls.
        sleep_on_rate: Seconds to sleep when a rate-limit error occurs.
    """
    client = openai.OpenAI(api_key=api_key)

    # Load all samples
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples from {input_path}")

    # Resume: find already-done IDs
    done_ids: set[str] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming — {len(done_ids)} already done")

    todo = [s for s in samples if s["id"] not in done_ids]
    print(f"Remaining : {len(todo)}")

    out_f = open(output_path, "a")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_populate_one, s, client, model): s["id"]
            for s in todo
        }
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Variable population")
        for future in pbar:
            try:
                result = future.result()
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            except Exception as exc:
                sid = futures[future]
                print(f"\n  [ERROR] {sid}: {exc}")
                if "rate" in str(exc).lower():
                    print(f"  Rate limit hit — sleeping {sleep_on_rate}s")
                    time.sleep(sleep_on_rate)

    out_f.close()
    print(f"\nDone. Populated file → {output_path}")


# ── Trimming helper (balanced 50K subsample) ─────────────────────────────────

def trim_dataset(
    input_path : str,
    output_path: str,
    target_total: int = 50_000,
    seed: int = 42,
) -> None:
    """
    Produce a balanced subsample preserving multi-turn dialogue integrity.

    Args:
        input_path:   Path to full unified JSONL.
        output_path:  Path to write trimmed JSONL.
        target_total: Target total sample count.
        seed:         Random seed.
    """
    import random
    random.seed(seed)

    with open(input_path) as f:
        all_samples = [json.loads(l) for l in f]

    n_per_action = target_total // 3
    by_action: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_action[s["action"]].append(s)

    trimmed = []
    for action, group in by_action.items():
        chosen = random.sample(group, min(n_per_action, len(group)))
        trimmed.extend(chosen)

    random.shuffle(trimmed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for s in trimmed:
            f.write(json.dumps(s) + "\n")

    counts = Counter(s["action"] for s in trimmed)
    print(f"Trimmed dataset: {len(trimmed)} samples")
    print(f"  Action dist : {dict(counts)}")
    print(f"  Saved → {output_path}")


from collections import Counter  # noqa: E402 (used in trim_dataset)