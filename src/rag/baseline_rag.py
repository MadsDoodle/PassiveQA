"""
Baseline RAG Pipeline for PassiveQA.

Steps:
    1. Load unified dataset → source-aware chunking → KB
    2. Embed chunks with all-MiniLM-L6-v2 → FAISS index
    3. Retrieve top-k chunks by cosine similarity
    4. Build Mistral-7B-Instruct prompt → generate ACTION + RESPONSE
    5. Evaluate on balanced held-out set
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from typing import Any, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig)


# ── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_WORD_LIMIT = 350
CHUNK_OVERLAP    = 50


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_ctx_text(sample: dict) -> str:
    docs = sample["context"]["documents"]
    return " ".join(d["text"].strip() for d in docs if d.get("text", "").strip())


def _chunk_text(text: str, max_words: int = CHUNK_WORD_LIMIT,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split long text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _needs_chunking(text: str, source: str) -> bool:
    wc = len(text.split())
    return (source == "contract_nli" and wc > 300) or \
           (source == "quac" and wc > 400)


# ── KB construction ──────────────────────────────────────────────────────────

def build_kb(all_samples: list[dict]) -> list[dict]:
    """
    Build a flat knowledge base from unified dataset samples.

    Returns:
        List of KB entry dicts, each with kb_id, text, source, etc.
    """
    kb_entries   = []
    chunk_id_ctr = 0
    seen_chunks: dict[str, int] = {}

    ctx_groups: dict[str, list] = defaultdict(list)
    for s in all_samples:
        ctx_text = _get_ctx_text(s)
        if ctx_text.strip():
            ctx_groups[_hash_text(ctx_text)].append(s)

    print(f"Unique contexts before chunking: {len(ctx_groups)}")

    for ctx_hash, samples in ctx_groups.items():
        ctx_text       = _get_ctx_text(samples[0])
        source         = samples[0]["metadata"]["source"]
        docs           = samples[0]["context"]["documents"]
        linked_ids     = [s["id"]     for s in samples]
        linked_actions = [s["action"] for s in samples]
        linked_queries = [s["query"]  for s in samples]
        action_dist    = dict(Counter(linked_actions))

        chunks = (_chunk_text(ctx_text) if _needs_chunking(ctx_text, source)
                  else [ctx_text])

        for chunk_idx, chunk in enumerate(chunks):
            ch = _hash_text(chunk)
            if ch in seen_chunks:
                continue

            doc_meta: dict = {}
            if source == "quac":
                doc_meta = {"dialogue_id": samples[0]["metadata"].get("dialogue_id", "")}
            elif source == "sharc":
                doc_meta = {"url": docs[0].get("url", "") if docs else ""}
            elif source == "contract_nli":
                doc_meta = {
                    "doc_id": docs[0].get("doc_id", "") if docs else "",
                    "file_name": docs[0].get("file_name", "") if docs else "",
                    "chunk_idx": chunk_idx, "total_chunks": len(chunks),
                }
            elif source == "hotpotqa":
                doc_meta = {"supporting_titles": [d.get("doc_id", "") for d in docs]}

            entry = {
                "kb_id"              : f"kb_{chunk_id_ctr:07d}",
                "chunk_hash"         : ch,
                "parent_ctx_hash"    : ctx_hash,
                "chunk_idx"          : chunk_idx,
                "total_chunks"       : len(chunks),
                "source"             : source,
                "text"               : chunk,
                "word_count"         : len(chunk.split()),
                "doc_meta"           : doc_meta,
                "linked_sample_ids"  : linked_ids,
                "linked_actions"     : linked_actions,
                "linked_queries"     : linked_queries,
                "action_distribution": action_dist,
                "num_linked"         : len(linked_ids),
            }
            seen_chunks[ch] = chunk_id_ctr
            kb_entries.append(entry)
            chunk_id_ctr += 1

    src_counts = Counter(e["source"] for e in kb_entries)
    print(f"Total KB chunks: {len(kb_entries)}")
    for src, cnt in src_counts.items():
        avg_w = np.mean([e["word_count"] for e in kb_entries if e["source"] == src])
        print(f"  {src:15}: {cnt:6} chunks | avg {avg_w:.0f} words")
    return kb_entries


def save_kb(kb_entries: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(kb_entries, f, indent=2)
    print(f"Saved KB → {path}")


# ── Embedding + FAISS ────────────────────────────────────────────────────────

def embed_kb(kb_entries: list[dict],
             emb_path: str,
             faiss_path: str,
             id_map_path: str,
             model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """
    Embed KB entries and build a FAISS index.

    Returns:
        (embedder, faiss_index, kb_id_map)
    """
    embedder = SentenceTransformer(model_name)
    texts    = [e["text"] for e in kb_entries]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = embedder.encode(
        texts, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(emb_path, embeddings)
    faiss.write_index(index, faiss_path)

    kb_id_map = {e["kb_id"]: i for i, e in enumerate(kb_entries)}
    with open(id_map_path, "w") as f:
        json.dump(kb_id_map, f)

    print(f"FAISS index: {index.ntotal} vectors | dim={dim}")
    return embedder, index, kb_id_map


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, embedder, index, kb_entries: list[dict],
             top_k: int = 5,
             source_filter: Optional[list[str]] = None) -> list[dict]:
    """Retrieve top-k chunks for a query via cosine similarity."""
    q_emb = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True)
    search_k = top_k * 5 if source_filter else top_k
    scores, indices = index.search(q_emb, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(kb_entries):
            continue
        entry = kb_entries[idx]
        if source_filter and entry["source"] not in source_filter:
            continue
        results.append({
            "kb_id"         : entry["kb_id"],
            "source"        : entry["source"],
            "score"         : round(float(score), 4),
            "text"          : entry["text"],
            "action_dist"   : entry["action_distribution"],
            "linked_queries": entry["linked_queries"][:3],
            "word_count"    : entry["word_count"],
        })
        if len(results) == top_k:
            break
    return results


# ── LLM prompt + generation ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a decision-aware intelligent assistant.

Given a user query and retrieved context passages, you must:

1. Carefully read the context.
2. Decide which action is appropriate:
   - ANSWER   → if the context contains sufficient information to answer the query
   - ASK      → if key information is missing and a clarification question would help
   - ABSTAIN  → if the query cannot be answered from the context at all

3. Output your response in this EXACT format:
ACTION: <ANSWER|ASK|ABSTAIN>
RESPONSE: <your answer, clarification question, or abstain statement>

Rules:
- Never guess or hallucinate. If uncertain → ASK or ABSTAIN.
- If ASKing, ask ONE focused clarification question.
- If ABSTAINing, say exactly why you cannot answer.
- Be concise.\
"""


def load_model(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
               load_4bit: bool = True):
    """Load Mistral-7B-Instruct (optionally 4-bit quantized)."""
    bnb_config = None
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded {model_id}")
    return tokenizer, model


def build_prompt(query: str, retrieved_chunks: list[dict],
                 tokenizer, history: Optional[list] = None) -> str:
    ctx_block = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        ctx_block += (f"\n[Context {i} | source={chunk['source']} "
                      f"| score={chunk['score']}]\n")
        ctx_block += chunk["text"][:800] + "\n"

    history_block = ""
    if history:
        history_block = "Conversation so far:\n"
        for turn in history:
            history_block += f"  User: {turn['query']}\n"
            history_block += f"  Assistant [{turn['action']}]: {turn['response']}\n"
        history_block += "\n"

    user_content = (f"{history_block}Query:\n{query}\n\n"
                    f"Retrieved Context:\n{ctx_block}\n")
    messages = [{"role": "user",
                 "content": SYSTEM_PROMPT + "\n\n" + user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def rag_infer(query: str, embedder, index, kb_entries: list[dict],
              tokenizer, model,
              top_k: int = 5,
              source_filter: Optional[list[str]] = None,
              history: Optional[list] = None) -> dict:
    """Full baseline RAG inference: retrieve → prompt → Mistral → parse."""
    chunks     = retrieve(query, embedder, index, kb_entries,
                          top_k=top_k, source_filter=source_filter)
    prompt_str = build_prompt(query, chunks, tokenizer, history)

    inputs = tokenizer(prompt_str, return_tensors="pt",
                       truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    action   = "ABSTAIN"
    resp_txt = raw_output
    for line in raw_output.split("\n"):
        ls = line.strip()
        if ls.startswith("ACTION:"):
            a = ls.replace("ACTION:", "").strip()
            if a in {"ANSWER", "ASK", "ABSTAIN"}:
                action = a
        if ls.startswith("RESPONSE:"):
            resp_txt = ls.replace("RESPONSE:", "").strip()

    return {
        "query"           : query,
        "action"          : action,
        "response"        : resp_txt,
        "raw_output"      : raw_output,
        "retrieved_chunks": chunks,
        "num_retrieved"   : len(chunks),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def sample_eval_set(all_samples: list[dict],
                    n_per_action: int = 100,
                    seed: int = 42) -> list[dict]:
    random.seed(seed)
    by_action: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_action[s["action"]].append(s)
    eval_set = []
    for action, samples in by_action.items():
        eval_set.extend(random.sample(samples, min(n_per_action, len(samples))))
    random.shuffle(eval_set)
    return eval_set


def evaluate_rag(eval_samples: list[dict],
                 embedder, index, kb_entries, tokenizer, model,
                 top_k: int = 5,
                 output_path: str = "eval_results.json") -> list[dict]:
    """Run baseline RAG eval and save results."""
    results, correct, hallucin = [], 0, 0
    pred_actions, gt_actions   = [], []

    for i, sample in enumerate(eval_samples):
        if i % 20 == 0:
            print(f"  {i+1}/{len(eval_samples)}...")
        try:
            out = rag_infer(sample["query"], embedder, index, kb_entries,
                            tokenizer, model, top_k=top_k)
        except Exception as ex:
            print(f"  Error on {i}: {ex}")
            out = {"action": "ABSTAIN", "response": "error",
                   "retrieved_chunks": [], "raw_output": ""}

        gt_action   = sample["action"]
        pred_action = out["action"]
        pred_actions.append(pred_action)
        gt_actions.append(gt_action)

        is_correct = (pred_action == gt_action)
        if is_correct: correct += 1
        if pred_action == "ANSWER" and gt_action != "ANSWER": hallucin += 1

        results.append({
            "sample_id"    : sample["id"],
            "source"       : sample["metadata"]["source"],
            "query"        : sample["query"],
            "gt_action"    : gt_action,
            "pred_action"  : pred_action,
            "response"     : out["response"],
            "correct"      : is_correct,
            "top_retrieved": [{"source": c["source"], "score": c["score"]}
                              for c in out["retrieved_chunks"]],
        })

    n = len(results)
    print("\n" + "=" * 55)
    print("  BASELINE RAG RESULTS")
    print("=" * 55)
    print(f"  Decision Accuracy  : {correct/n*100:.1f}%")
    print(f"  Hallucination Rate : {hallucin/n*100:.1f}%")
    for act in ["ANSWER", "ASK", "ABSTAIN"]:
        ar = [r for r in results if r["gt_action"] == act]
        if ar:
            ac = sum(r["correct"] for r in ar)
            print(f"  {act:10}: {ac}/{len(ar)} ({100*ac/len(ar):.1f}%)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": {"decision_accuracy": round(correct/n, 4),
                               "hallucination_rate": round(hallucin/n, 4),
                               "n_evaluated": n},
                   "results": results}, f, indent=2)
    print(f"  Saved → {output_path}")
    return results