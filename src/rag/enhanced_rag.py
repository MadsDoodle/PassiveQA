"""
Enhanced RAG Pipeline for PassiveQA.

Improvements over baseline:
    A. Semantic chunking (sentence-boundary aware, with overlap)
    B. Multi-granularity KB (coarse chunks + fine sentences)
    C. Hybrid retrieval (BM25 + dense, score fusion)
    D. Query understanding (rewrite + multi-hop decomposition)
    E. Cross-encoder re-ranking + context compression
    F. Full pipeline with self-reflection
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Optional

import faiss
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ── A. Semantic chunking ──────────────────────────────────────────────────────

def semantic_chunk(text: str, max_words: int = 350,
                   overlap_sents: int = 2) -> list[str]:
    """
    Chunk text at sentence boundaries with sentence-level overlap.

    Args:
        text:          Input text.
        max_words:     Maximum words per chunk.
        overlap_sents: Number of trailing sentences to carry into next chunk.

    Returns:
        List of chunk strings.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current_sents: list[str] = []
    current_wc = 0

    for sent in sentences:
        sent_wc = len(sent.split())
        if current_wc + sent_wc > max_words and current_sents:
            chunks.append(" ".join(current_sents))
            current_sents = current_sents[-overlap_sents:]
            current_wc    = sum(len(s.split()) for s in current_sents)
        current_sents.append(sent)
        current_wc += sent_wc

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks if chunks else [text]


def _needs_chunking(text: str, source: str) -> bool:
    wc = len(text.split())
    return (source == "contract_nli" and wc > 300) or \
           (source == "quac" and wc > 400)


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_ctx_text(sample: dict) -> str:
    docs = sample["context"]["documents"]
    return " ".join(d["text"].strip() for d in docs if d.get("text", "").strip())


# ── B. Multi-granularity KB ───────────────────────────────────────────────────

def build_mg_kb(all_samples: list[dict],
                save_path: Optional[str] = None) -> list[dict]:
    """
    Build a two-level knowledge base:
        - coarse: full semantic chunks
        - fine: individual sentences (min 5 words)

    Args:
        all_samples: Unified dataset samples.
        save_path:   Optional path to cache the KB as JSON.

    Returns:
        List of multi-granularity KB entry dicts.
    """
    if save_path and os.path.exists(save_path):
        print(f"Loading MG-KB from cache: {save_path}")
        with open(save_path) as f:
            return json.load(f)

    ctx_groups: dict[str, list] = defaultdict(list)
    for s in all_samples:
        ctx_text = _get_ctx_text(s)
        if ctx_text.strip():
            ctx_groups[_hash_text(ctx_text)].append(s)

    mg_kb: list[dict] = []
    seen: dict[str, int] = {}
    entry_id = 0

    for ctx_hash, samples in ctx_groups.items():
        ctx_text       = _get_ctx_text(samples[0])
        source         = samples[0]["metadata"]["source"]
        linked_ids     = [s["id"]     for s in samples]
        linked_actions = [s["action"] for s in samples]
        linked_queries = [s["query"]  for s in samples]
        action_dist    = dict(Counter(linked_actions))

        coarse_chunks = (semantic_chunk(ctx_text)
                         if _needs_chunking(ctx_text, source)
                         else [ctx_text])

        for c_idx, chunk in enumerate(coarse_chunks):
            ch = _hash_text(chunk)
            if ch not in seen:
                seen[ch] = entry_id
                mg_kb.append({
                    "mg_id": f"mg_{entry_id:07d}", "granularity": "coarse",
                    "parent_ctx_hash": ctx_hash, "chunk_idx": c_idx,
                    "source": source, "text": chunk,
                    "word_count": len(chunk.split()),
                    "linked_sample_ids": linked_ids,
                    "linked_actions": linked_actions,
                    "linked_queries": linked_queries,
                    "action_distribution": action_dist,
                })
                entry_id += 1

        for s_idx, sent in enumerate(sent_tokenize(ctx_text)):
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue
            sh = _hash_text(sent)
            if sh not in seen:
                seen[sh] = entry_id
                mg_kb.append({
                    "mg_id": f"mg_{entry_id:07d}", "granularity": "fine",
                    "parent_ctx_hash": ctx_hash, "sent_idx": s_idx,
                    "source": source, "text": sent,
                    "word_count": len(sent.split()),
                    "linked_sample_ids": linked_ids,
                    "linked_actions": linked_actions,
                    "linked_queries": linked_queries,
                    "action_distribution": action_dist,
                })
                entry_id += 1

    coarse = sum(1 for e in mg_kb if e["granularity"] == "coarse")
    fine   = sum(1 for e in mg_kb if e["granularity"] == "fine")
    print(f"MG-KB: {len(mg_kb)} entries | coarse={coarse} | fine={fine}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(mg_kb, f, indent=2)
        print(f"Saved MG-KB → {save_path}")

    return mg_kb


def embed_mg_kb(mg_kb: list[dict],
                emb_path: str, faiss_path: str,
                model_name: str = "all-MiniLM-L6-v2"):
    """Embed the multi-granularity KB and build a FAISS index."""
    if (os.path.exists(emb_path) and os.path.exists(faiss_path)):
        print("Loading cached MG embeddings + FAISS index...")
        mg_embeddings = np.load(emb_path)
        mg_index      = faiss.read_index(faiss_path)
        embedder      = SentenceTransformer(model_name)
        return embedder, mg_index, mg_embeddings

    embedder = SentenceTransformer(model_name)
    texts    = [e["text"] for e in mg_kb]
    mg_embeddings = embedder.encode(
        texts, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    dim      = mg_embeddings.shape[1]
    mg_index = faiss.IndexFlatIP(dim)
    mg_index.add(mg_embeddings)

    np.save(emb_path, mg_embeddings)
    faiss.write_index(mg_index, faiss_path)
    print(f"MG FAISS: {mg_index.ntotal} vectors | dim={dim}")
    return embedder, mg_index, mg_embeddings


def build_bm25(mg_kb: list[dict]):
    """Build a BM25 index over the MG-KB."""
    corpus = [e["text"].lower().split() for e in mg_kb]
    return BM25Okapi(corpus)


# ── C. Hybrid retrieval ───────────────────────────────────────────────────────

def hybrid_retrieve(query: str, mg_kb: list[dict],
                    embedder, mg_index, bm25_index,
                    top_k: int = 5, alpha: float = 0.5,
                    granularity_filter: Optional[str] = None) -> list[dict]:
    """
    Hybrid BM25 + dense retrieval with score fusion.

    Args:
        query:               User query string.
        alpha:               Dense weight (0=pure BM25, 1=pure dense).
        granularity_filter:  'coarse' | 'fine' | None.

    Returns:
        Top-k retrieved entries with fused scores.
    """
    n = len(mg_kb)

    # Dense scores
    q_emb = embedder.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True)
    dense_scores, dense_idxs = mg_index.search(q_emb, min(n, 200))
    dense_score_map = {int(i): float(s)
                       for i, s in zip(dense_idxs[0], dense_scores[0])
                       if i >= 0}

    # BM25 scores
    bm25_raw  = bm25_index.get_scores(query.lower().split())
    bm25_max  = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
    bm25_norm = bm25_raw / bm25_max

    # Candidate pool: union of top-200 from each method
    bm25_top200 = np.argsort(bm25_raw)[::-1][:200]
    candidates  = set(dense_score_map.keys()) | set(bm25_top200.tolist())

    scored = []
    for idx in candidates:
        if idx >= n:
            continue
        entry = mg_kb[idx]
        if granularity_filter and entry["granularity"] != granularity_filter:
            continue
        d_score = dense_score_map.get(idx, 0.0)
        b_score = float(bm25_norm[idx])
        fused   = alpha * d_score + (1 - alpha) * b_score
        scored.append((fused, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for fused_score, idx in scored[:top_k]:
        e = mg_kb[idx]
        results.append({
            "mg_id"         : e["mg_id"],
            "granularity"   : e["granularity"],
            "source"        : e["source"],
            "fused_score"   : round(fused_score, 4),
            "dense_score"   : round(dense_score_map.get(idx, 0.0), 4),
            "bm25_score"    : round(float(bm25_norm[idx]), 4),
            "text"          : e["text"],
            "action_dist"   : e["action_distribution"],
            "linked_queries": e["linked_queries"][:2],
            "word_count"    : e["word_count"],
        })
    return results


# ── D. Query understanding ────────────────────────────────────────────────────

def rewrite_query(query: str, tokenizer, model) -> str:
    """Rewrite a vague query for better retrieval using Mistral."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          f"Rewrite the following query to be more specific and retrieval-friendly.\n"
          f"Keep it as ONE concise sentence. Do not answer it.\n"
          f"Return ONLY the rewritten query, nothing else.\n\n"
          f"Original query: {query}\nRewritten query:"}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def decompose_query(query: str, tokenizer, model) -> list[str]:
    """Decompose a multi-hop query into simpler sub-queries."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content":
          f"Decompose the following complex question into 2-3 simpler sub-questions "
          f"that can be answered independently.\nReturn ONLY the sub-questions, "
          f"one per line, numbered.\nIf the question is already simple, "
          f"return just: 1. {query}\n\nQuestion: {query}\nSub-questions:"}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    sub_qs = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            sub_q = line.split(".", 1)[-1].strip()
            if sub_q:
                sub_qs.append(sub_q)
    return sub_qs if sub_qs else [query]


def is_multihop(query: str) -> bool:
    keywords = {"and", "both", "compare", "difference",
                "before", "after", "when did", "which of"}
    q_lower  = query.lower()
    return (len(query.split()) > 12 or
            any(k in q_lower for k in keywords))


# ── E. Re-ranking + compression ───────────────────────────────────────────────

def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    print(f"Loading cross-encoder: {model_name}")
    return CrossEncoder(model_name)


def rerank(query: str, chunks: list[dict],
           reranker: CrossEncoder, top_n: int = 5) -> list[dict]:
    """Re-rank retrieved chunks using a cross-encoder."""
    if not chunks:
        return chunks
    pairs  = [(query, c["text"][:512]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_n]:
        chunk["rerank_score"] = round(float(score), 4)
        result.append(chunk)
    return result


def compress_context(query: str, chunks: list[dict],
                     max_sents_per_chunk: int = 3) -> list[dict]:
    """Keep only the most relevant sentences from each chunk."""
    query_words = query.lower().split()
    compressed  = []
    for chunk in chunks:
        sentences = sent_tokenize(chunk["text"])
        if len(sentences) <= max_sents_per_chunk:
            compressed.append(chunk)
            continue
        sent_scores = [(sum(1 for w in query_words if w in s.lower().split()), s)
                       for s in sentences]
        top_texts = {s for _, s in
                     sorted(sent_scores, key=lambda x: x[0], reverse=True)[:max_sents_per_chunk]}
        ordered = [s for s in sentences if s in top_texts]
        c = dict(chunk)
        c["text"] = " ".join(ordered)
        c["compressed"] = True
        compressed.append(c)
    return compressed


# ── F. Full enhanced pipeline ─────────────────────────────────────────────────

SYSTEM_PROMPT_ENHANCED = """\
You are a decision-aware intelligent assistant.

Given a user query and retrieved context passages, you must:

1. Carefully read the context.
2. Decide which action is appropriate:
   - ANSWER   → sufficient information present
   - ASK      → key information missing, clarification needed
   - ABSTAIN  → cannot be answered from context at all

3. Output in this EXACT format:
ACTION: <ANSWER|ASK|ABSTAIN>
RESPONSE: <answer, clarification question, or abstain statement>

Rules:
- Never hallucinate. If uncertain → ASK or ABSTAIN.
- If ASKing, ask ONE focused clarification question.
- Be concise.\
"""

REFLECTION_PROMPT = """\
You generated this response to a query.
Verify if your action is correct given the context.

Query: {query}
Your action: {action}
Your response: {response}
Context used: {context_preview}

Is your action correct?
- If ANSWER: is it supported by context? If not → change to ABSTAIN
- If ASK: is clarification truly needed? If context is sufficient → change to ANSWER
- If ABSTAIN: is context truly insufficient?

Output ONLY:
VERIFIED_ACTION: <ANSWER|ASK|ABSTAIN>
VERIFIED_RESPONSE: <corrected response if needed, else same>\
"""


def _llm_generate(prompt: str, tokenizer, model,
                  max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    new_toks = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


def _parse_action_response(raw: str,
                            action_key: str = "ACTION:",
                            response_key: str = "RESPONSE:") -> tuple[str, str]:
    action   = "ABSTAIN"
    resp_txt = raw
    for line in raw.split("\n"):
        ls = line.strip()
        if ls.startswith(action_key):
            a = ls.replace(action_key, "").strip()
            if a in {"ANSWER", "ASK", "ABSTAIN"}:
                action = a
        if ls.startswith(response_key):
            resp_txt = ls.replace(response_key, "").strip()
    return action, resp_txt


def enhanced_rag_infer(
    query: str,
    mg_kb: list[dict],
    embedder, mg_index, bm25_index, reranker,
    tokenizer, model,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5,
    alpha: float = 0.5,
    history: Optional[list] = None,
    use_rewrite: bool = True,
    use_decompose: bool = True,
    use_reflection: bool = True,
) -> dict:
    """
    Full enhanced RAG pipeline:
    rewrite → decompose → hybrid retrieve → rerank → compress → generate → reflect
    """
    trace = {"original_query": query}

    working_query = query
    if use_rewrite:
        working_query       = rewrite_query(query, tokenizer, model)
        trace["rewritten_q"] = working_query

    all_chunks: list[dict] = []
    if use_decompose and is_multihop(query):
        sub_qs              = decompose_query(query, tokenizer, model)
        trace["sub_queries"] = sub_qs
        for sq in sub_qs:
            all_chunks.extend(
                hybrid_retrieve(sq, mg_kb, embedder, mg_index, bm25_index,
                                top_k=top_k_retrieve // max(len(sub_qs), 1),
                                alpha=alpha)
            )
        # Deduplicate
        seen_ids: set[str] = set()
        all_chunks = [c for c in all_chunks
                      if not (c["mg_id"] in seen_ids or seen_ids.add(c["mg_id"]))]
    else:
        all_chunks = hybrid_retrieve(working_query, mg_kb, embedder, mg_index,
                                     bm25_index, top_k=top_k_retrieve, alpha=alpha)

    trace["num_retrieved"] = len(all_chunks)

    reranked   = rerank(working_query, all_chunks, reranker, top_n=top_k_rerank)
    compressed = compress_context(working_query, reranked)

    # Build prompt and generate
    ctx_block = ""
    for i, c in enumerate(compressed, 1):
        score = c.get("rerank_score", c.get("fused_score", "—"))
        ctx_block += (f"\n[Context {i} | src={c['source']} "
                      f"| gran={c.get('granularity', '—')} "
                      f"| score={score}]\n")
        ctx_block += c["text"][:600] + "\n"

    history_block = ""
    if history:
        history_block = "Conversation so far:\n"
        for t in history:
            history_block += f"  User: {t['query']}\n"
            history_block += f"  Assistant [{t['action']}]: {t['response']}\n"
        history_block += "\n"

    content  = (f"{history_block}Query:\n{query}\n\n"
                f"Retrieved Context:\n{ctx_block}")
    messages = [{"role": "user",
                 "content": SYSTEM_PROMPT_ENHANCED + "\n\n" + content}]
    prompt   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    raw_output        = _llm_generate(prompt, tokenizer, model)
    action, response  = _parse_action_response(raw_output)
    trace["first_action"]   = action
    trace["first_response"] = response

    if use_reflection:
        ctx_preview = " | ".join(c["text"][:100] for c in compressed[:2])
        ref_content = REFLECTION_PROMPT.format(
            query=query, action=action,
            response=response, context_preview=ctx_preview)
        ref_prompt  = tokenizer.apply_chat_template(
            [{"role": "user", "content": ref_content}],
            tokenize=False, add_generation_prompt=True)
        ref_raw     = _llm_generate(ref_prompt, tokenizer, model,
                                    max_new_tokens=150)
        v_action, v_response = _parse_action_response(
            ref_raw, action_key="VERIFIED_ACTION:",
            response_key="VERIFIED_RESPONSE:")
        trace["reflection_changed"] = (v_action != action)
        if v_action != action:
            action, response = v_action, v_response

    trace["final_action"]   = action
    trace["final_response"] = response

    return {
        "query"            : query,
        "action"           : action,
        "response"         : response,
        "raw_output"       : raw_output,
        "retrieved_chunks" : compressed,
        "num_retrieved"    : len(compressed),
        "trace"            : trace,
    }