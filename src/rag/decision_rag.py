"""
Decision-Aware RAG v3 for PassiveQA.

Architecture:
    Section 1: Evidence scoring (confidence, coverage, ambiguity, conflict)
    Section 2: Answerability classifier (dedicated LLM call)
    Section 3: Hard-gate decision module (rule-based, no casual deciding)
    Section 4: Action-specific generators (separate prompt per action)
    Section 5: Full v3 pipeline
    Section 6: Evaluation
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder, SentenceTransformer


# ── Section 1: Evidence scoring ───────────────────────────────────────────────

def compute_evidence_scores(query: str, chunks: list[dict]) -> dict:
    """
    Compute three retrieval-quality signals without an LLM call.

    Returns dict with: confidence_score, coverage_score, score_gap,
                       mean_score, num_chunks
    """
    if not chunks:
        return {"confidence_score": 0.0, "coverage_score": 0.0,
                "score_gap": 0.0, "mean_score": 0.0, "num_chunks": 0}

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    raw_scores  = [c.get("rerank_score", c.get("fused_score", 0.0))
                   for c in chunks]
    norm_scores = [sigmoid(s) if (s > 1 or s < 0) else s for s in raw_scores]

    confidence = float(np.max(norm_scores))
    mean_score = float(np.mean(norm_scores))
    score_gap  = (float(norm_scores[0] - norm_scores[1])
                  if len(norm_scores) > 1 else confidence)

    # Coverage: fraction of content query words present in retrieved chunks
    stop_words    = {"is","the","a","an","of","in","and","or","to",
                     "it","that","this","for","on","at"}
    query_words   = set(query.lower().split()) - stop_words
    if not query_words:
        coverage = confidence
    else:
        all_text = " ".join(c["text"].lower() for c in chunks)
        covered  = sum(1 for w in query_words if w in all_text)
        coverage = covered / len(query_words)

    return {
        "confidence_score": round(confidence, 4),
        "coverage_score"  : round(coverage, 4),
        "score_gap"       : round(score_gap, 4),
        "mean_score"      : round(mean_score, 4),
        "num_chunks"      : len(chunks),
    }


def compute_ambiguity_score(query: str) -> float:
    """
    Heuristic ambiguity detection — no LLM needed.

    Returns a score in [0, 1] where 1 = very ambiguous.
    """
    signals: list[float] = []
    q_lower = query.lower().strip()
    q_words = q_lower.split()

    # 1. Very short queries are likely underspecified
    signals.append(1.0 if len(q_words) <= 4 else 0.0)

    # 2. Dangling pronouns
    pronouns = {"it","its","they","their","this","that","these",
                "those","he","she","him","her"}
    signals.append(0.8 if any(w in pronouns for w in q_words) else 0.0)

    # 3. Vague quantifiers
    vague = {"something","anything","everything","some","any",
             "many","various","certain","related"}
    signals.append(0.6 if any(w in vague for w in q_words) else 0.0)

    # 4. No proper noun / named entity signal
    question_starters = {"can","does","is","are","what","who","when",
                         "where","why","how"}
    has_caps = any(w[0].isupper() for w in query.split()
                   if len(w) > 2 and w.lower() not in question_starters)
    signals.append(0.4 if not has_caps else 0.0)

    # 5. Comparative without both sides
    comp_kw = {"better","worse","difference","compare","vs","versus","between"}
    if any(w in comp_kw for w in q_words):
        cap_ents = [w for w in query.split() if w[0].isupper()]
        signals.append(0.7 if len(cap_ents) < 2 else 0.0)
    else:
        signals.append(0.0)

    return round(float(np.mean(signals)), 4)


def check_context_conflict(chunks: list[dict], embedder) -> float:
    """
    Detect contradicting chunks via pairwise cosine similarity.

    Low average similarity between top chunks = high conflict score.
    Returns conflict_score in [0, 1].
    """
    if len(chunks) < 2:
        return 0.0
    texts = [c["text"][:300] for c in chunks[:4]]
    embs  = embedder.encode(texts, normalize_embeddings=True,
                             convert_to_numpy=True)
    sims  = [float(np.dot(embs[i], embs[j]))
             for i in range(len(embs))
             for j in range(i + 1, len(embs))]
    return round(max(0.0, 1.0 - float(np.mean(sims))), 4)


# ── Section 2: Answerability classifier ──────────────────────────────────────

ANSWERABILITY_PROMPT = """\
You are an evidence evaluator.

Given a query and retrieved context, evaluate ONLY whether the context
contains sufficient information.

Output EXACTLY one of:
ANSWERABLE        - context clearly supports answering the query
NEEDS_CLARIFICATION - query is ambiguous or missing key information
NOT_ANSWERABLE    - context lacks the information needed

Then output one line:
REASON: <one short sentence why>

Do NOT answer the query. Only evaluate answerability.

Query: {query}

Context:
{context}

Output:\
"""


def answerability_classify(query: str, chunks: list[dict],
                            tokenizer, model) -> dict:
    """
    Dedicated LLM call just for decision — no answer generation.

    Returns dict with: label, reason, raw_output
    """
    ctx = ""
    for i, c in enumerate(chunks[:3], 1):
        ctx += f"[{i}] {c['text'][:400]}\n"

    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": ANSWERABILITY_PROMPT.format(query=query, context=ctx)}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1536).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    new_toks   = out[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_toks, skip_special_tokens=True).strip()

    label  = "NOT_ANSWERABLE"
    reason = ""
    for line in raw_output.split("\n"):
        ls = line.strip()
        if "ANSWERABLE" in ls and "NOT" not in ls and "NEEDS" not in ls:
            label = "ANSWERABLE"
        elif "NEEDS_CLARIFICATION" in ls:
            label = "NEEDS_CLARIFICATION"
        elif "NOT_ANSWERABLE" in ls:
            label = "NOT_ANSWERABLE"
        if ls.startswith("REASON:"):
            reason = ls.replace("REASON:", "").strip()

    return {"label": label, "reason": reason, "raw_output": raw_output}


# ── Section 3: Hard-gate decision module ─────────────────────────────────────

# Tunable thresholds
TAU_CONFIDENCE = 0.35   # below → ABSTAIN
TAU_COVERAGE   = 0.30   # below → ABSTAIN
TAU_AMBIGUITY  = 0.45   # above → ASK
TAU_CONFLICT   = 0.70   # above → ASK or ABSTAIN


def hard_gate_decision(
    ev_scores: dict, ambiguity_score: float,
    conflict_score: float, answerability_label: str,
    tau_confidence: float = TAU_CONFIDENCE,
    tau_coverage:   float = TAU_COVERAGE,
    tau_ambiguity:  float = TAU_AMBIGUITY,
    tau_conflict:   float = TAU_CONFLICT,
) -> tuple[str, str]:
    """
    Rule-based hard-gate decision.  No LLM casual deciding.

    Returns:
        (action, reason_string)
    """
    conf     = ev_scores["confidence_score"]
    coverage = ev_scores["coverage_score"]
    n_chunks = ev_scores["num_chunks"]

    # 1. Empty retrieval → ABSTAIN
    if n_chunks == 0:
        return "ABSTAIN", "no_context_retrieved"

    # 2. LLM answerability = NOT_ANSWERABLE and low confidence → ABSTAIN
    if answerability_label == "NOT_ANSWERABLE" and conf < tau_confidence:
        return "ABSTAIN", "low_confidence_not_answerable"

    # 3. High conflict and ambiguous → ASK
    if conflict_score >= tau_conflict and ambiguity_score >= tau_ambiguity:
        return "ASK", "high_conflict_and_ambiguity"

    # 4. Very low coverage → ABSTAIN
    if coverage < tau_coverage and n_chunks > 0:
        return "ABSTAIN", "low_coverage"

    # 5. Ambiguous query → ASK
    if ambiguity_score >= tau_ambiguity:
        return "ASK", "ambiguous_query"

    # 6. LLM says NEEDS_CLARIFICATION → ASK
    if answerability_label == "NEEDS_CLARIFICATION":
        return "ASK", "llm_needs_clarification"

    # 7. Low confidence → ABSTAIN
    if conf < tau_confidence:
        return "ABSTAIN", "low_confidence"

    return "ANSWER", "sufficient_evidence"


# ── Section 4: Action-specific generators ────────────────────────────────────

ANSWER_PROMPT = """\
You are a precise factual assistant.

The query has been verified as answerable from the context below.
Answer DIRECTLY and CONCISELY using only the provided context.
Do not add information not present in the context.

Query: {query}

Context:
{context}

Answer:\
"""

ASK_PROMPT = """\
You are a clarification assistant.

The query cannot be fully answered because key information is missing.
Generate ONE focused clarification question that would resolve the gap.

Query: {query}

Context (partial):
{context}

Reason clarification is needed: {reason}

Clarification question:\
"""

ABSTAIN_PROMPT = """\
You are an honest assistant.

You cannot answer the query because the context is insufficient.
Explain briefly why you cannot answer and what information is missing.

Query: {query}

Reason: {reason}

Response:\
"""


def _generate_with_prompt(prompt: str, tokenizer, model,
                           max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_toks = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


def generate_answer(query: str, chunks: list[dict],
                    tokenizer, model) -> str:
    ctx = "\n".join(f"[{i}] {c['text'][:500]}"
                    for i, c in enumerate(chunks[:5], 1))
    return _generate_with_prompt(
        ANSWER_PROMPT.format(query=query, context=ctx),
        tokenizer, model)


def generate_ask(query: str, chunks: list[dict],
                 reason: str, tokenizer, model) -> str:
    ctx = "\n".join(f"[{i}] {c['text'][:300]}"
                    for i, c in enumerate(chunks[:3], 1))
    return _generate_with_prompt(
        ASK_PROMPT.format(query=query, context=ctx, reason=reason),
        tokenizer, model)


def generate_abstain(query: str, reason: str,
                     tokenizer, model) -> str:
    return _generate_with_prompt(
        ABSTAIN_PROMPT.format(query=query, reason=reason),
        tokenizer, model)


# ── Section 5: Full v3 pipeline ───────────────────────────────────────────────

def decision_aware_rag(
    query: str,
    mg_kb: list[dict],
    embedder, mg_index, bm25_index, reranker,
    tokenizer, model,
    top_k_retrieve: int = 20,
    top_k_rerank:   int = 5,
    alpha: float = 0.5,
    history: Optional[list] = None,
    use_rewrite: bool = True,
    use_decompose: bool = True,
) -> dict:
    """
    Decision-Aware RAG v3 pipeline.

    Flow: retrieve → score → classify → hard-gate → action-specific generate
    """
    # Import enhanced helpers to avoid circular imports
    from .enhanced_rag import (hybrid_retrieve, rerank, compress_context,
                                rewrite_query, decompose_query, is_multihop)

    trace = {"original_query": query}

    # 1. Query rewrite
    working_query = query
    if use_rewrite:
        working_query       = rewrite_query(query, tokenizer, model)
        trace["rewritten_q"] = working_query

    # 2. Retrieve
    all_chunks: list[dict] = []
    if use_decompose and is_multihop(query):
        sub_qs               = decompose_query(query, tokenizer, model)
        trace["sub_queries"] = sub_qs
        for sq in sub_qs:
            all_chunks.extend(
                hybrid_retrieve(sq, mg_kb, embedder, mg_index, bm25_index,
                                top_k=top_k_retrieve // max(len(sub_qs), 1),
                                alpha=alpha))
        seen_ids: set = set()
        all_chunks = [c for c in all_chunks
                      if not (c["mg_id"] in seen_ids
                              or seen_ids.add(c["mg_id"]))]
    else:
        all_chunks = hybrid_retrieve(working_query, mg_kb, embedder,
                                     mg_index, bm25_index,
                                     top_k=top_k_retrieve, alpha=alpha)

    # 3. Rerank + compress
    reranked   = rerank(working_query, all_chunks, reranker, top_n=top_k_rerank)
    compressed = compress_context(working_query, reranked)

    # 4. Evidence scoring
    ev_scores       = compute_evidence_scores(query, compressed)
    ambiguity_score = compute_ambiguity_score(query)
    conflict_score  = check_context_conflict(compressed, embedder)
    trace["evidence_scores"] = ev_scores
    trace["ambiguity_score"] = ambiguity_score
    trace["conflict_score"]  = conflict_score

    # 5. Answerability classification (dedicated LLM call)
    ans_result = answerability_classify(query, compressed, tokenizer, model)
    trace["answerability"] = ans_result["label"]

    # 6. Hard-gate decision
    action, reason = hard_gate_decision(
        ev_scores, ambiguity_score, conflict_score, ans_result["label"])
    trace["gate_reason"] = reason

    # 7. Action-specific generation
    if action == "ANSWER":
        response = generate_answer(query, compressed, tokenizer, model)
    elif action == "ASK":
        response = generate_ask(query, compressed, reason, tokenizer, model)
    else:
        response = generate_abstain(query, reason, tokenizer, model)

    trace["final_action"]   = action
    trace["final_response"] = response

    return {
        "query"            : query,
        "action"           : action,
        "response"         : response,
        "retrieved_chunks" : compressed,
        "num_retrieved"    : len(compressed),
        "trace"            : trace,
    }