"""
Evaluation for the PassiveQA Decision Planner.

Loads the finetuned LoRA adapter + Mistral-7B, runs inference on
the held-out test split, and reports per-class F1, confusion matrix,
per-source accuracy, and multi-turn vs single-turn accuracy.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch


ACTIONS = ["ANSWER", "ASK", "ABSTAIN"]

SYSTEM_PROMPT = """You are a decision planner for a question-answering system.

Your task: given a user query, search the knowledge graph for relevant nodes, evaluate what information is present and what is missing, then decide the correct action.

Decision logic:
- Search the graph for nodes matching the query subject and known variables
- If the graph contains a complete path connecting known entities to an answer → ANSWER
- If the graph contains the topic but key linking variables are missing → ASK (specify what is missing)
- If the graph has no relevant nodes or the topic is entirely absent → ABSTAIN

Output format (strictly follow this):
<reasoning>
Step 1 — Query subject: identify what the query is asking about
Step 2 — Graph search: what nodes were found, what connections exist
Step 3 — Variable check: what is known, what is missing
Step 4 — Decision rationale: why this action is correct
</reasoning>

<decision>
ANSWER | ASK | ABSTAIN
</decision>

<justification>
One sentence grounded in the graph evidence.
</justification>"""


def load_model_and_tokenizer(
    adapter_dir:  str,
    model_id:     str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_len:  int = 1024,
):
    """
    Load the base model + LoRA adapter for evaluation.
    Returns (model, tokenizer, use_bf16, max_seq_len).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    use_bf16  = torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype       = torch.bfloat16 if use_bf16 else torch.float16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    alloc = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✅ Model ready  |  VRAM: {alloc:.1f} GB")
    return model, tokenizer, use_bf16, max_seq_len


def _extract_decision(text: str) -> str:
    if "<decision>" in text and "</decision>" in text:
        s = text.find("<decision>") + len("<decision>")
        e = text.find("</decision>")
        return text[s:e].strip()
    return "UNKNOWN"


def _extract_true_label(sample: dict) -> str:
    return _extract_decision(sample["messages"][2]["content"])


def _extract_metadata(sample: dict) -> tuple[str, bool]:
    raw = json.dumps(sample).lower()
    src = next(
        (s for s in ["quac", "sharc", "hotpotqa", "contract_nli"] if s in raw),
        "unknown",
    )
    is_mt = "<conversation_history>" in sample["messages"][1]["content"]
    return src, is_mt


def predict(
    sample:      dict,
    model,
    tokenizer,
    max_seq_len: int = 1024,
) -> tuple[str, str]:
    """Run the planner on a single test sample. Returns (decision, full_response)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        sample["messages"][1],   # original user turn
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=max_seq_len,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 400,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    gen  = out[0][inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(gen, skip_special_tokens=True).strip()
    dec  = _extract_decision(resp)
    return (dec if dec in ACTIONS else "UNKNOWN"), resp


def run_evaluation(
    test_file:    str,
    model,
    tokenizer,
    max_seq_len:  int = 1024,
    eval_per_class: int = 30,
    plot_dir:     Optional[str] = None,
    metrics_path: Optional[str] = None,
    random_seed:  int = 42,
) -> dict:
    """
    Run balanced evaluation on the test set.

    Args:
        test_file:      Path to ft_test_mistral.jsonl.
        model:          Loaded LoRA model (eval mode).
        tokenizer:      Matching tokenizer.
        max_seq_len:    Max tokens for inference.
        eval_per_class: Samples per action class (balanced evaluation).
        plot_dir:       Directory to save eval plots (None = skip plots).
        metrics_path:   Path to save eval_metrics.json (None = skip).
        random_seed:    Random seed for balanced sampling.

    Returns:
        Dict containing all metrics.
    """
    from tqdm import tqdm

    random.seed(random_seed)
    np.random.seed(random_seed)

    # ── Load test set ─────────────────────────────────────────────
    test_samples = []
    with open(test_file) as f:
        for line in f:
            test_samples.append(json.loads(line))
    print(f"  Total test samples  : {len(test_samples)}")

    # Balanced sample
    by_class: dict = defaultdict(list)
    for s in test_samples:
        by_class[_extract_true_label(s)].append(s)

    eval_samples = []
    for action in ACTIONS:
        pool = by_class.get(action, [])
        random.shuffle(pool)
        eval_samples.extend(pool[:eval_per_class])
    random.shuffle(eval_samples)
    print(f"  Evaluating          : {len(eval_samples)} samples "
          f"({eval_per_class} per class)")

    # ── Inference ────────────────────────────────────────────────
    y_true:    list[str]  = []
    y_pred:    list[str]  = []
    sources:   list[str]  = []
    is_mt_arr: list[bool] = []

    print("\nRunning inference …")
    for s in tqdm(eval_samples, desc="Eval"):
        true_label          = _extract_true_label(s)
        pred_label, _       = predict(s, model, tokenizer, max_seq_len)
        src, is_mt          = _extract_metadata(s)

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label in ACTIONS else "UNKNOWN")
        sources.append(src)
        is_mt_arr.append(is_mt)

    # ── Metrics ───────────────────────────────────────────────────
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix, f1_score
        )
        acc         = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        macro_f1    = f1_score(y_true, y_pred, labels=ACTIONS,
                               average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=ACTIONS,
                               average="weighted", zero_division=0)
        per_f1      = f1_score(y_true, y_pred, labels=ACTIONS,
                               average=None,       zero_division=0)
        unknown_ct  = sum(1 for p in y_pred if p == "UNKNOWN")
        cm          = confusion_matrix(y_true, y_pred, labels=ACTIONS)

        print(f"\n{'='*60}")
        print(f"  Accuracy    : {acc*100:.2f}%")
        print(f"  Macro F1    : {macro_f1*100:.2f}%")
        print(f"  Weighted F1 : {weighted_f1*100:.2f}%")
        print(f"  Unparseable : {unknown_ct} "
              f"({100*unknown_ct/len(y_pred):.1f}%)")
        print()
        print(classification_report(y_true, y_pred,
                                    labels=ACTIONS, zero_division=0))

    except ImportError:
        # sklearn not available: basic counts only
        acc      = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        macro_f1 = weighted_f1 = acc
        per_f1   = np.zeros(3)
        cm_list  = []
        unknown_ct = sum(1 for p in y_pred if p == "UNKNOWN")
        print(f"  Accuracy: {acc*100:.2f}%  (install scikit-learn for full metrics)")

    # per-source accuracy
    src_res: dict = defaultdict(lambda: {"c": 0, "t": 0})
    for t, p, src in zip(y_true, y_pred, sources):
        src_res[src]["t"] += 1
        src_res[src]["c"] += int(t == p)
    print("  Per-source accuracy:")
    for src, r in sorted(src_res.items()):
        print(f"    {src:15}: {r['c']/max(r['t'],1)*100:.1f}%  "
              f"({r['c']}/{r['t']})")

    # turn-type accuracy
    mt_c = sum(t == p for t, p, m in zip(y_true, y_pred, is_mt_arr) if m)
    mt_t = sum(1 for m in is_mt_arr if m)
    st_c = sum(t == p for t, p, m in zip(y_true, y_pred, is_mt_arr) if not m)
    st_t = sum(1 for m in is_mt_arr if not m)
    print(f"\n  Multi-turn  : {mt_c/max(mt_t,1)*100:.1f}%  ({mt_c}/{mt_t})")
    print(f"  Single-turn : {st_c/max(st_t,1)*100:.1f}%  ({st_c}/{st_t})")

    metrics = {
        "accuracy"        : round(float(acc), 4),
        "macro_f1"        : round(float(macro_f1), 4),
        "weighted_f1"     : round(float(weighted_f1), 4),
        "per_class_f1"    : {a: round(float(f), 4)
                             for a, f in zip(ACTIONS, per_f1)},
        "per_source_acc"  : {s: round(r["c"] / max(r["t"], 1), 4)
                             for s, r in src_res.items()},
        "multiturn_acc"   : round(mt_c / max(mt_t, 1), 4),
        "singleturn_acc"  : round(st_c / max(st_t, 1), 4),
        "unknown_preds"   : unknown_ct,
        "eval_n"          : len(eval_samples),
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else [],
    }

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✅ Metrics → {metrics_path}")

    if plot_dir:
        _save_plots(metrics, per_f1, cm, ACTIONS, plot_dir, acc, macro_f1,
                    len(eval_samples), src_res)

    print(f"\n{'='*60}")
    print(f"  PAPER SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy     : {acc*100:.1f}%")
    print(f"  Macro F1     : {macro_f1*100:.1f}%")
    for a, f in zip(ACTIONS, per_f1):
        print(f"  {a:10} F1 : {f*100:.1f}%")
    print(f"  Multi-turn   : {mt_c/max(mt_t,1)*100:.1f}%")
    print(f"  Single-turn  : {st_c/max(st_t,1)*100:.1f}%")
    print(f"{'='*60}")

    return metrics


def _save_plots(
    metrics:  dict,
    per_f1,
    cm,
    actions:  list,
    plot_dir: str,
    acc:      float,
    macro_f1: float,
    n:        int,
    src_res:  dict,
) -> None:
    """Save confusion matrix, per-class F1, and per-source accuracy plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(actions, color="white")
    ax.set_yticks(range(3)); ax.set_yticklabels(actions, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True",      color="white")
    ax.set_title("Confusion Matrix", color="white", fontsize=13)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)

    # Per-class F1
    ax = axes[1]
    bars = ax.bar(actions, per_f1 * 100,
                  color=["#57CC99", "#4E9AF1", "#FF6B6B"], width=0.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("F1 Score (%)", color="white")
    ax.set_title("Per-class F1 Score", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for bar, val in zip(bars, per_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f"{val*100:.1f}%",
                ha="center", color="white", fontsize=11, fontweight="bold")
    ax.axhline(y=macro_f1 * 100, color="#FFD166", linestyle="--",
               linewidth=1.5, label=f"Macro F1: {macro_f1*100:.1f}%")
    ax.legend(fontsize=9, labelcolor="white",
              facecolor="#2a2a3e", edgecolor="#444")

    # Per-source accuracy
    ax        = axes[2]
    src_names = sorted(src_res.keys())
    src_accs  = [src_res[s]["c"] / max(src_res[s]["t"], 1) * 100
                 for s in src_names]
    bars = ax.bar(src_names, src_accs,
                  color=["#F4A261", "#9B72CF", "#57CC99", "#4E9AF1"][:len(src_names)],
                  width=0.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Accuracy by Source", color="white", fontsize=13)
    ax.tick_params(colors="white", axis="both")
    for lbl in ax.get_xticklabels():
        lbl.set_color("white"); lbl.set_fontsize(9)
    for bar, val in zip(bars, src_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f"{val:.1f}%",
                ha="center", color="white", fontsize=10, fontweight="bold")
    ax.axhline(y=acc * 100, color="#FFD166", linestyle="--",
               linewidth=1.5, label=f"Overall: {acc*100:.1f}%")
    ax.legend(fontsize=9, labelcolor="white",
              facecolor="#2a2a3e", edgecolor="#444")

    plt.suptitle(
        f"Planner Evaluation — Mistral-7B LoRA  |  "
        f"Acc: {acc*100:.1f}%  Macro-F1: {macro_f1*100:.1f}%  n={n}",
        color="white", fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "eval_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Plot → {plot_path}")