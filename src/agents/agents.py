"""
Three-agent response generators for PassiveQA.

Each agent is called after the planner has made its routing decision:
  ANSWER  → answer_agent  — RAG-grounded factual answer
  ASK     → ask_agent     — focused clarification question
  ABSTAIN → abstain_agent — honest, informative refusal
"""

from __future__ import annotations

from typing import Optional


def answer_agent(
    query:       str,
    history:     Optional[list]  = None,
    # full-pipeline parameters (used when enhanced RAG is available)
    top_k_retrieve: int = 20,
    top_k_rerank:   int = 5,
    alpha:          float = 0.5,
    # lightweight stub parameters (used in synthetic / no-RAG mode)
    graph_triples: Optional[list] = None,
    kb_text:       str = "",
    # model / tokenizer injected at call time
    model=None,
    tokenizer=None,
    max_seq_len:   int = 1024,
    # enhanced-RAG helpers (optional, provided by pipeline)
    retrieve_fn=None,
    rerank_fn=None,
    compress_fn=None,
    rewrite_fn=None,
    decompose_fn=None,
    is_multihop_fn=None,
) -> dict:
    """
    Generate a RAG-grounded answer.

    When the full enhanced-RAG helpers are provided (retrieve_fn etc.) the
    agent runs a complete hybrid-retrieval + rerank + compress pipeline.
    Otherwise it falls back to the lightweight stub that uses pre-supplied
    graph_triples and kb_text directly.

    Returns:
        {"action": "ANSWER", "response": str, "sources": list, ...}
    """
    # ── Full RAG path ──────────────────────────────────────────────
    if retrieve_fn is not None:
        working_query = rewrite_fn(query) if rewrite_fn else query

        if is_multihop_fn and is_multihop_fn(query) and decompose_fn:
            sub_qs     = decompose_fn(query)
            all_chunks = []
            per_sub    = max(2, top_k_retrieve // len(sub_qs))
            for sq in sub_qs:
                all_chunks.extend(retrieve_fn(sq, top_k=per_sub, alpha=alpha))
            seen_ids, dedup = set(), []
            for c in all_chunks:
                if c["mg_id"] not in seen_ids:
                    seen_ids.add(c["mg_id"])
                    dedup.append(c)
            all_chunks = dedup
        else:
            all_chunks = retrieve_fn(
                working_query, top_k=top_k_retrieve, alpha=alpha
            )

        reranked   = rerank_fn(working_query, all_chunks, top_n=top_k_rerank) \
                     if rerank_fn else all_chunks[:top_k_rerank]
        compressed = compress_fn(working_query, reranked) \
                     if compress_fn else reranked

        ctx_block = ""
        for i, c in enumerate(compressed, 1):
            ctx_block += (
                f"\n[Source {i} | {c['source']} | "
                f"{c.get('granularity','—')}]\n"
                f"{c['text'][:500]}\n"
            )

        history_block = _format_history(history)
        prompt_text = (
            f"You are a knowledgeable assistant. Answer the query using ONLY "
            f"the provided context. Be concise and factual.\n\n"
            f"{history_block}Query: {query}\n\nContext:{ctx_block}\n\nAnswer:"
        )
        answer  = _generate(prompt_text, model, tokenizer,
                             max_new_tokens=300, max_seq_len=max_seq_len)
        sources = [{"source": c["source"], "text_preview": c["text"][:80]}
                   for c in compressed]
        return {
            "action"          : "ANSWER",
            "response"        : answer,
            "sources"         : sources,
            "retrieved_chunks": len(compressed),
            "rewritten_query" : working_query,
        }

    # ── Lightweight stub path ──────────────────────────────────────
    chunks: list[dict] = []
    if kb_text:
        chunks.append({
            "mg_id"      : "syn_chunk_0",
            "granularity": "coarse",
            "source"     : "kb_text",
            "text"       : kb_text,
        })
    if graph_triples:
        chunks.append({
            "mg_id"      : "syn_chunk_kg",
            "granularity": "fine",
            "source"     : "kg_triples",
            "text"       : "\n".join(graph_triples),
        })

    ctx_block = ""
    for i, c in enumerate(chunks, 1):
        ctx_block += (
            f"\n[Source {i} | {c['source']} | "
            f"{c.get('granularity','—')}]\n"
            f"{c['text'][:500]}\n"
        )

    history_block = _format_history(history)
    prompt_text = (
        f"You are a knowledgeable assistant. Answer the query using ONLY "
        f"the provided context. Be concise and factual.\n\n"
        f"{history_block}Query: {query}\n\nContext:{ctx_block}\n\nAnswer:"
    )
    answer = _generate(prompt_text, model, tokenizer,
                       max_new_tokens=200, max_seq_len=max_seq_len)
    return {
        "action"  : "ANSWER",
        "response": answer,
        "sources" : [{"source": c["source"], "text_preview": c["text"][:80]}
                     for c in chunks],
    }


def ask_agent(
    query:             str,
    missing_variables: list,
    known_variables:   list,
    graph_triples:     Optional[list] = None,
    history:           Optional[list] = None,
    model=None,
    tokenizer=None,
    max_seq_len:       int = 1024,
) -> dict:
    """
    Generate a focused clarification question.

    Returns:
        {"action": "ASK", "response": str, "missing": list, "anchor": str|None}
    """
    missing_str = (
        ", ".join(f"'{m}'" for m in missing_variables[:2])
        if missing_variables else "specific details"
    )

    graph_entities = []
    for t in (graph_triples or [])[:5]:
        parts = t.split(" | ")
        if len(parts) == 3 and not parts[0].startswith("?"):
            graph_entities.append(parts[0].strip())
    graph_entities = list(dict.fromkeys(graph_entities))[:3]

    anchor = (
        graph_entities[0] if graph_entities
        else (known_variables[0] if known_variables else None)
    )

    history_block = ""
    if history:
        history_block = "Previous turns:\n"
        for h in history[-3:]:
            history_block += (
                f"  User: {h['query']}\n"
                f"  Assistant [{h['action']}]: {h['response']}\n"
            )
        history_block += "\n"

    prompt_text = (
        f"Ask ONE focused clarification question to help answer the user's query.\n\n"
        f"{history_block}User query: {query}\n"
        f"Missing information: {missing_str}\n"
        f"Known context: {', '.join(graph_entities) if graph_entities else 'none'}\n\n"
        f"Rules: Ask exactly ONE question ending with ?\n"
        f"Be specific about what is missing. Reference the query topic.\n\n"
        f"Clarification question:"
    )
    question = _generate(prompt_text, model, tokenizer,
                         max_new_tokens=80, max_seq_len=max_seq_len)
    question = question.split("\n")[0].strip()

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    if not question or len(question) < 10:
        if anchor and missing_variables:
            question = f"Regarding {anchor}, could you clarify {missing_variables[0]}?"
        elif missing_variables:
            question = f"Could you provide more details about {missing_variables[0]}?"
        else:
            question = "Could you provide more specific details about your query?"

    return {
        "action"  : "ASK",
        "response": question,
        "missing" : missing_variables,
        "anchor"  : anchor,
    }


def abstain_agent(
    query:             str,
    known_variables:   list,
    graph_triples:     Optional[list] = None,
    missing_variables: Optional[list] = None,
    history:           Optional[list] = None,
    model=None,
    tokenizer=None,
    max_seq_len:       int = 1024,
) -> dict:
    """
    Generate an honest, informative refusal.

    Returns:
        {"action": "ABSTAIN", "response": str, "reason": str}
    """
    has_graph  = len(graph_triples  or []) > 0
    has_known  = len(known_variables or []) > 0
    is_vague   = len(query.split()) < 5

    if is_vague and not has_known:
        reason = "The query is too vague to match any information in the knowledge base."
    elif has_graph and not has_known:
        reason = (
            "The knowledge base contains related content but "
            "it does not connect to the specific question asked."
        )
    elif missing_variables:
        reason = (
            f"The required information — "
            f"{', '.join((missing_variables or [])[:2])} — "
            f"is entirely absent from the knowledge base."
        )
    else:
        reason = "The topic of this query is not covered in the available knowledge base."

    history_note = ""
    if history:
        resolved = [h["resolved_variable"] for h in history
                    if h.get("resolved_variable")]
        if resolved:
            history_note = (
                f"Already clarified in this conversation: "
                f"{', '.join(resolved)}. "
            )

    prompt_text = (
        f"You cannot answer the following query from the available knowledge base.\n"
        f"Write a brief, honest refusal. Explain why you cannot answer.\n"
        f"Do NOT make up information.\n\n"
        f"{history_note}Query: {query}\n"
        f"Reason: {reason}\n\n"
        f"Your response:"
    )
    refusal = _generate(prompt_text, model, tokenizer,
                        max_new_tokens=100, max_seq_len=max_seq_len)

    if not refusal or len(refusal) < 20:
        refusal = (
            f"I'm unable to answer this query. {reason} "
            f"You may want to consult a specialised source."
        )

    return {
        "action"  : "ABSTAIN",
        "response": refusal,
        "reason"  : reason,
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _format_history(history: Optional[list]) -> str:
    if not history:
        return ""
    block = "Conversation so far:\n"
    for h in history[-3:]:
        block += (
            f"  User: {h['query']}\n"
            f"  Assistant: {h['response']}\n"
        )
    return block + "\n"


def _generate(
    prompt_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
    max_seq_len:    int = 1024,
) -> str:
    """Call model.generate on a raw prompt string."""
    import torch

    if model is None or tokenizer is None:
        return "[model not loaded]"

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        formatted, return_tensors="pt",
        truncation=True, max_length=max_seq_len,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()