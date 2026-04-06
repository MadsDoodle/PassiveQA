"""
PassiveQA Full Inference Pipeline.

Planner (finetuned Mistral-7B LoRA)
  → routes to ANSWER / ASK / ABSTAIN agent
  → returns structured result dict

Usage (after loading model + tokenizer):
    from src.agents.pipeline import run_pipeline

    result = run_pipeline(
        query             = "Am I eligible for the pension plan?",
        known_variables   = ["pension plan"],
        graph_triples     = ["pension plan | require | years of service",
                             "pension plan | requires | ?unknown_1"],
        missing_variables = ["years of service", "employment type"],
    )
    print(result["action"], result["response"])
"""

from __future__ import annotations

from typing import Optional

import torch

from .agents import answer_agent, ask_agent, abstain_agent

# ── System prompt (shared between planner and evaluation) ─────────────────────

SYSTEM_PROMPT = """You are a decision planner for a question-answering system.

Your task: given a user query, search the knowledge graph for relevant nodes, evaluate what information is present and what is missing, then decide the correct action.

Decision logic:
- Search the graph for nodes matching the query subject and known variables
- If the graph contains a complete path connecting known entities to an answer → ANSWER
- If the graph contains the topic but key linking variables are missing → ASK (specify what is missing)
- If the graph has no relevant nodes or the topic is entirely absent → ABSTAIN

You will receive:
<query> — the user's question
<known_variables> — entities explicitly present in the query
<graph_context> — KG triples from nodes matching the query (subject | relation | object)
<missing_variables> — variables required but not present
<conversation_history> — prior turns (for multi-turn queries only)

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
</justification>

Rules:
- Reasoning must reference actual graph content, not generic statements
- Never say "unspecified variables" — name the specific missing variable
- If graph_context is empty, default to ABSTAIN unless context is clearly partial (then ASK)
- Do not use prior world knowledge — only the graph context provided"""

VALID_DECISIONS = {"ANSWER", "ASK", "ABSTAIN"}


# ── Planner ───────────────────────────────────────────────────────────────────

def run_planner(
    query:             str,
    known_variables:   list[str],
    graph_triples:     list[str],
    missing_variables: list[str],
    history:           Optional[list] = None,
    model=None,
    tokenizer=None,
    max_seq_len:       int = 1024,
    stopping_criteria=None,
) -> tuple[str, str]:
    """
    Run the finetuned planner model.

    Returns:
        (decision, full_response)
        decision ∈ {"ANSWER", "ASK", "ABSTAIN"}
    """
    graph_block = (
        "<graph_context>\n"
        + "\n".join(graph_triples)
        + "\n</graph_context>"
    ) if graph_triples else (
        "<graph_context>\nNo relevant nodes found in knowledge graph.\n</graph_context>"
    )

    missing_block = (
        "<missing_variables>\n"
        + "\n".join(f"- {m}" for m in missing_variables)
        + "\n</missing_variables>"
    ) if missing_variables else "<missing_variables>\nnone\n</missing_variables>"

    history_block = ""
    if history:
        history_block = "<conversation_history>\n"
        for h in history:
            history_block += (
                f"Turn {h['turn_id']} | {h['action']} | "
                f"Q: \"{h['query'][:80]}\" | A: {h['response']}\n"
            )
            if h.get("resolved_variable"):
                history_block += f"  → resolved: '{h['resolved_variable']}'\n"
        history_block += "</conversation_history>\n\n"

    user_content = (
        f"{history_block}"
        f"<query>\n{query}\n</query>\n\n"
        f"<known_variables>\n"
        f"{', '.join(known_variables) if known_variables else 'none identified'}\n"
        f"</known_variables>\n\n"
        f"{graph_block}\n\n"
        f"{missing_block}\n\n"
        f"Search the graph context for relevant nodes and decide the correct action."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    if model is None or tokenizer is None:
        raise RuntimeError(
            "run_planner requires `model` and `tokenizer` arguments. "
            "Load them with load_model_and_tokenizer() first."
        )

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt",
        truncation=True, max_length=max_seq_len,
    ).to(model.device)

    gen_kwargs: dict = dict(
        max_new_tokens = 400,
        do_sample      = False,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    decision = "ABSTAIN"
    if "<decision>" in resp and "</decision>" in resp:
        s        = resp.find("<decision>") + len("<decision>")
        e        = resp.find("</decision>")
        decision = resp[s:e].strip()

    if decision not in VALID_DECISIONS:
        decision = "ABSTAIN"

    return decision, resp


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    query:             str,
    known_variables:   Optional[list[str]] = None,
    graph_triples:     Optional[list[str]] = None,
    missing_variables: Optional[list[str]] = None,
    history:           Optional[list]      = None,
    # model injection
    model=None,
    tokenizer=None,
    max_seq_len:       int = 1024,
    stopping_criteria=None,
    # optional full-RAG helpers
    retrieve_fn=None,
    rerank_fn=None,
    compress_fn=None,
    rewrite_fn=None,
    decompose_fn=None,
    is_multihop_fn=None,
    # optional KG auto-retrieval
    kg_cache=None,
    # lightweight stub (synthetic mode)
    kb_text:           str = "",
    verbose:           bool = True,
) -> dict:
    """
    Full PassiveQA inference pipeline.

    1. (Optional) Auto-retrieve KG triples if kg_cache is provided.
    2. Run planner → ANSWER / ASK / ABSTAIN decision.
    3. Route to the corresponding agent.
    4. Return structured result dict.

    Args:
        query:             User's natural-language query.
        known_variables:   Entities already known from the query.
        graph_triples:     Pre-retrieved KG triples (skip if kg_cache provided).
        missing_variables: Variables identified as missing.
        history:           Prior conversation turns (multi-turn mode).
        model:             LoRA-adapted model (must be provided).
        tokenizer:         Matching tokenizer (must be provided).
        max_seq_len:       Max tokens for all generation calls.
        stopping_criteria: Optional HF StoppingCriteriaList.
        retrieve_fn:       (query, top_k, alpha) → list[chunk] (enhanced RAG).
        rerank_fn:         (query, chunks, top_n) → list[chunk].
        compress_fn:       (query, chunks) → list[chunk].
        rewrite_fn:        (query) → str.
        decompose_fn:      (query) → list[str].
        is_multihop_fn:    (query) → bool.
        kg_cache:          KGCache object for auto-retrieval.
        kb_text:           Raw KB text for lightweight answer generation.
        verbose:           Print progress to stdout.

    Returns:
        {
          "action":           str,          # ANSWER / ASK / ABSTAIN
          "response":         str,          # agent output
          "planner_reasoning": str,         # full planner output
          "graph_triples":    list[str],
          # agent-specific keys: "sources", "missing", "reason", ...
        }
    """
    known_variables   = known_variables   or []
    graph_triples     = graph_triples     or []
    missing_variables = missing_variables or []

    # ── Auto-retrieve KG context ──────────────────────────────────
    if kg_cache is not None and not graph_triples:
        from ..finetune.dataset_creation import (
            get_subgraph_triples, cap_requires_edges, infer_effective_missing,
        )
        retrieved, _, _ = get_subgraph_triples(
            query, known_variables, missing_variables, "ASK", kg_cache
        )
        graph_triples     = cap_requires_edges(retrieved, action="ASK")
        missing_variables = infer_effective_missing(
            query, known_variables, missing_variables,
            graph_triples, "ASK", kg_cache,
        )

    if verbose:
        print(f"\n  Query   : {query}")
        print(f"  Known   : {known_variables}")
        print(f"  Triples : {len(graph_triples)}")

    # ── Planner decision ──────────────────────────────────────────
    decision, planner_response = run_planner(
        query             = query,
        known_variables   = known_variables,
        graph_triples     = graph_triples,
        missing_variables = missing_variables,
        history           = history,
        model             = model,
        tokenizer         = tokenizer,
        max_seq_len       = max_seq_len,
        stopping_criteria = stopping_criteria,
    )

    if verbose:
        print(f"  Decision: {decision}")

    # ── Route to agent ────────────────────────────────────────────
    agent_kwargs = dict(
        model=model, tokenizer=tokenizer, max_seq_len=max_seq_len
    )

    if decision == "ANSWER":
        result = answer_agent(
            query         = query,
            history       = history,
            graph_triples = graph_triples,
            kb_text       = kb_text,
            retrieve_fn   = retrieve_fn,
            rerank_fn     = rerank_fn,
            compress_fn   = compress_fn,
            rewrite_fn    = rewrite_fn,
            decompose_fn  = decompose_fn,
            is_multihop_fn= is_multihop_fn,
            **agent_kwargs,
        )
    elif decision == "ASK":
        result = ask_agent(
            query             = query,
            missing_variables = missing_variables,
            known_variables   = known_variables,
            graph_triples     = graph_triples,
            history           = history,
            **agent_kwargs,
        )
    else:   # ABSTAIN
        result = abstain_agent(
            query             = query,
            known_variables   = known_variables,
            graph_triples     = graph_triples,
            missing_variables = missing_variables,
            history           = history,
            **agent_kwargs,
        )

    result["planner_reasoning"] = planner_response
    result["graph_triples"]     = graph_triples

    if verbose:
        print(f"  Response: {result['response'][:120]}")

    return result