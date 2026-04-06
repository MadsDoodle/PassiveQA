"""
Unified Dataset Construction for PassiveQA.

Converts four QA benchmarks into a single decision-aware schema with
explicit ANSWER / ASK / ABSTAIN labels.

Sources:
    QuAC        → CANNOTANSWER → ABSTAIN, followup=y → ASK, else → ANSWER
    ShARC       → Yes/No → ANSWER, Follow-on → ASK, Irrelevant → ABSTAIN
    HotPotQA    → ANSWER only (multi-hop reasoning)
    ContractNLI → Entailment/Contradiction → ANSWER, NotMentioned → ABSTAIN
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from typing import Any


# ── Schema helpers ──────────────────────────────────────────────────────────

def _make_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{str(idx).zfill(6)}"


def _heuristic_difficulty(num_missing: int,
                           evidence_len: int = 0,
                           num_hops: int = 0) -> str:
    score = num_missing + evidence_len + num_hops
    if score == 0:    return "easy"
    elif score <= 2:  return "medium"
    elif score <= 4:  return "hard"
    else:             return "very_hard"


def _heuristic_failure_mode(action: str,
                             num_missing: int,
                             multi_hop: bool = False) -> str:
    if action == "ANSWER":             return "COMPLETE"
    if action == "ABSTAIN":            return "INSUFFICIENT_VARIABLES"
    if action == "ASK" and multi_hop:  return "MULTI_HOP_REQUIRED"
    if action == "ASK":                return "INSUFFICIENT_VARIABLES"
    return "COMPLETE"


def _heuristic_completeness(action: str) -> str:
    return {"ANSWER": "complete", "ASK": "partial",
            "ABSTAIN": "incomplete"}.get(action, "partial")


# ── Source builders ─────────────────────────────────────────────────────────

def build_quac_samples(quac_train: dict[str, Any]) -> list[dict]:
    """
    Build PassiveQA samples from a loaded QuAC train split.

    Args:
        quac_train: Parsed JSON of train_v0.2.json (QuAC format).

    Returns:
        List of unified sample dicts.
    """
    samples   = []
    sample_idx = 0

    for article in quac_train["data"]:
        for para in article["paragraphs"]:
            context_text = para.get("context", "")
            dialogue_id  = para.get("id", str(uuid.uuid4()))

            for turn_id, qa in enumerate(para["qas"]):
                ans_text  = (qa["answers"][0]["text"]
                             if qa["answers"] else "CANNOTANSWER")
                followup  = qa.get("followup", "n")
                is_cannot = (ans_text == "CANNOTANSWER")

                if is_cannot:
                    action   = "ABSTAIN"
                    response = "I do not have enough information to answer this question."
                elif followup == "y":
                    action   = "ASK"
                    response = "Could you provide more details so I can give a more precise answer?"
                else:
                    action   = "ANSWER"
                    response = ans_text

                num_missing  = 1 if action != "ANSWER" else 0
                failure_mode = _heuristic_failure_mode(action, num_missing)
                difficulty   = _heuristic_difficulty(num_missing)
                completeness = _heuristic_completeness(action)

                samples.append({
                    "id"     : _make_id("quac", sample_idx),
                    "query"  : qa["question"],
                    "context": {
                        "documents": [
                            {"doc_id": dialogue_id, "text": context_text}
                        ]
                    },
                    "state": {
                        "known_variables"  : [],
                        "missing_variables": [],
                        "constraints"      : [],
                        "failure_mode"     : failure_mode,
                        "difficulty"       : difficulty,
                        "completeness"     : completeness,
                    },
                    "action"  : action,
                    "response": response,
                    "metadata": {
                        "num_missing_variables": num_missing,
                        "variable_types"       : [],
                        "multi_turn"           : True,
                        "turn_id"              : turn_id + 1,
                        "dialogue_id"          : dialogue_id,
                        "requires_reasoning"   : False,
                        "source"               : "quac",
                        "yesno"                : qa.get("yesno", "x"),
                        "followup_flag"        : followup,
                    },
                })
                sample_idx += 1

    print(f"QuAC samples constructed : {len(samples)}")
    return samples


def build_sharc_samples(sharc_train: list[dict]) -> list[dict]:
    """
    Build PassiveQA samples from a loaded ShARC train split.

    Args:
        sharc_train: Parsed list from sharc_train.json.

    Returns:
        List of unified sample dicts.
    """
    samples    = []
    sample_idx = 0

    for item in sharc_train:
        answer   = item["answer"]
        evidence = item.get("evidence", [])
        history  = item.get("history", [])
        scenario = item.get("scenario", "").strip()

        if answer in ["Yes", "No"]:
            action   = "ANSWER"
            response = answer
        elif answer == "Follow-on":
            action   = "ASK"
            response = (evidence[0]["follow_up_question"]
                        if evidence else "Could you provide more details?")
        else:
            action   = "ABSTAIN"
            response = "I do not have enough information to determine this."

        num_missing  = (len(evidence) if action == "ASK"
                        else (1 if action == "ABSTAIN" else 0))
        failure_mode = _heuristic_failure_mode(action, num_missing)
        difficulty   = _heuristic_difficulty(num_missing, len(evidence),
                                              len(history))
        completeness = _heuristic_completeness(action)

        context_text = item.get("snippet", "")
        if scenario:
            context_text += f"\n\nUser Scenario: {scenario}"

        missing_vars = [ev["follow_up_question"] for ev in evidence]

        samples.append({
            "id"     : _make_id("sharc", sample_idx),
            "query"  : item["question"],
            "context": {
                "documents": [{
                    "doc_id": item["tree_id"],
                    "text"  : context_text,
                    "url"   : item.get("source_url", ""),
                }]
            },
            "state": {
                "known_variables"  : [],
                "missing_variables": missing_vars,
                "constraints"      : [],
                "failure_mode"     : failure_mode,
                "difficulty"       : difficulty,
                "completeness"     : completeness,
            },
            "action"  : action,
            "response": response,
            "metadata": {
                "num_missing_variables": num_missing,
                "variable_types"       : [],
                "multi_turn"           : len(history) > 0,
                "turn_id"              : None,
                "dialogue_id"          : item["tree_id"],
                "requires_reasoning"   : len(evidence) > 1,
                "source"               : "sharc",
                "utterance_id"         : item["utterance_id"],
                "sharc_answer"         : answer,
                "evidence_depth"       : len(evidence),
                "history_depth"        : len(history),
            },
        })
        sample_idx += 1

    print(f"ShARC samples constructed : {len(samples)}")
    return samples


def build_hotpotqa_samples(hotpot_train) -> list[dict]:
    """
    Build PassiveQA samples from a loaded HotPotQA train split.

    Args:
        hotpot_train: HuggingFace dataset split or list of dicts.

    Returns:
        List of unified sample dicts (all ANSWER).
    """
    samples    = []
    sample_idx = 0
    level_map  = {"easy": "easy", "medium": "medium", "hard": "hard"}

    for item in hotpot_train:
        sf_titles  = item["supporting_facts"]["title"]
        sf_sentids = item["supporting_facts"]["sent_id"]
        ctx_titles = item["context"]["title"]
        ctx_sents  = item["context"]["sentences"]

        title_to_sents = {t: s for t, s in zip(ctx_titles, ctx_sents)}

        supporting_texts = []
        for t, sid in zip(sf_titles, sf_sentids):
            if t in title_to_sents and sid < len(title_to_sents[t]):
                supporting_texts.append({
                    "doc_id": t,
                    "text"  : title_to_sents[t][sid],
                })

        difficulty = level_map.get(item.get("level", "medium"), "medium")

        samples.append({
            "id"     : _make_id("hotpot", sample_idx),
            "query"  : item["question"],
            "context": {"documents": supporting_texts},
            "state": {
                "known_variables"  : [],
                "missing_variables": [],
                "constraints"      : [],
                "failure_mode"     : "COMPLETE",
                "difficulty"       : difficulty,
                "completeness"     : "complete",
            },
            "action"  : "ANSWER",
            "response": item["answer"],
            "metadata": {
                "num_missing_variables": 0,
                "variable_types"       : [],
                "multi_turn"           : False,
                "turn_id"              : None,
                "dialogue_id"          : None,
                "requires_reasoning"   : item.get("type") == "bridge",
                "source"               : "hotpotqa",
                "question_type"        : item.get("type", ""),
                "level"                : item.get("level", ""),
                "num_supporting_facts" : len(sf_titles),
            },
        })
        sample_idx += 1

    print(f"HotPotQA samples constructed : {len(samples)}")
    return samples


def build_contractnli_samples(cnli_data: dict) -> list[dict]:
    """
    Build PassiveQA samples from loaded ContractNLI train data.

    Args:
        cnli_data: Parsed train.json from ContractNLI.

    Returns:
        List of unified sample dicts.
    """
    samples    = []
    sample_idx = 0
    labels     = cnli_data["labels"]
    documents  = cnli_data["documents"]

    for doc in documents:
        doc_text   = doc["text"]
        char_spans = doc["spans"]

        for annot_set in doc["annotation_sets"]:
            for label_id, annot in annot_set["annotations"].items():
                choice     = annot["choice"]
                span_idxs  = annot["spans"]
                hypothesis = labels[label_id]["hypothesis"]
                short_desc = labels[label_id]["short_description"]

                span_texts = []
                for idx in span_idxs:
                    if idx < len(char_spans):
                        s, e = char_spans[idx]
                        span_texts.append(doc_text[s:e].strip())

                if choice in ["Entailment", "Contradiction"]:
                    action   = "ANSWER"
                    response = (
                        f"Yes, the contract supports: {short_desc}."
                        if choice == "Entailment"
                        else f"No, the contract contradicts: {short_desc}."
                    )
                else:
                    action   = "ABSTAIN"
                    response = (f"The contract does not mention "
                                f"information related to: {short_desc}.")

                num_missing  = 0 if action == "ANSWER" else 1
                failure_mode = _heuristic_failure_mode(action, num_missing)
                difficulty   = _heuristic_difficulty(
                    num_missing, num_hops=len(span_idxs))
                completeness = _heuristic_completeness(action)

                samples.append({
                    "id"     : _make_id("cnli", sample_idx),
                    "query"  : hypothesis,
                    "context": {
                        "documents": [{
                            "doc_id"   : str(doc["id"]),
                            "file_name": doc.get("file_name", ""),
                            "text"     : doc_text,
                            "spans"    : span_texts,
                        }]
                    },
                    "state": {
                        "known_variables"  : [short_desc],
                        "missing_variables": ([] if action == "ANSWER"
                                              else [short_desc]),
                        "constraints"      : [],
                        "failure_mode"     : failure_mode,
                        "difficulty"       : difficulty,
                        "completeness"     : completeness,
                    },
                    "action"  : action,
                    "response": response,
                    "metadata": {
                        "num_missing_variables": num_missing,
                        "variable_types"       : ["constraint"],
                        "multi_turn"           : False,
                        "turn_id"              : None,
                        "dialogue_id"          : None,
                        "requires_reasoning"   : len(span_idxs) > 1,
                        "source"               : "contract_nli",
                        "label_id"             : label_id,
                        "nli_choice"           : choice,
                        "num_spans"            : len(span_idxs),
                    },
                })
                sample_idx += 1

    print(f"ContractNLI samples constructed : {len(samples)}")
    return samples


# ── Merge + save ────────────────────────────────────────────────────────────

def merge_and_save(quac_samples: list, sharc_samples: list,
                   hotpot_samples: list, cnli_samples: list,
                   output_dir: str) -> list[dict]:
    """
    Merge all source samples, print statistics, and save to disk.

    Outputs:
        - unified_train.jsonl   (full merged set)
        - {source}_train.jsonl  (per-source files for ablations)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_samples = (quac_samples + sharc_samples
                   + hotpot_samples + cnli_samples)

    action_counts = Counter(s["action"] for s in all_samples)
    source_counts = Counter(s["metadata"]["source"] for s in all_samples)

    print("\n" + "=" * 50)
    print(f"Total samples : {len(all_samples)}")
    print("\nAction distribution:")
    for k, v in action_counts.items():
        print(f"  {k:10} : {v:6}  ({100*v/len(all_samples):.1f}%)")
    print("\nSource distribution:")
    for k, v in source_counts.items():
        print(f"  {k:15} : {v}")

    # Save unified
    out_path = os.path.join(output_dir, "unified_train.jsonl")
    with open(out_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved → {out_path}")

    # Save per-source
    for name, samples in [("quac", quac_samples),
                           ("sharc", sharc_samples),
                           ("hotpotqa", hotpot_samples),
                           ("contract_nli", cnli_samples)]:
        path = os.path.join(output_dir, f"{name}_train.jsonl")
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {name:15} → {path}")

    return all_samples