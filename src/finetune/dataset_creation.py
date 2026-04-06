"""
Fine-tuning Dataset Creation for PassiveQA.

Converts the unified dataset + KG into supervised finetuning samples
in the Mistral chat format. Each sample is a 3-message sequence:
  system | user (query + KG context + variable state) | assistant (decision)

Decision labels: ANSWER / ASK / ABSTAIN
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TRIPLES        = 12
HOP_RADIUS         = 2
NODE_MATCH_THRESH  = 0.55
TRIPLE_REL_THRESH  = 0.35
MAX_VAR_SHOWN      = 3
SPECIFICITY_THRESHOLD = 0.45

GENERIC_ANCHORS = [
    "clarification needed",
    "more information required",
    "unspecified context",
    "further details",
    "additional context needed",
    "insufficient information",
]

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

DUMMY_RESPONSES = {
    "Could you provide more details so I can give a more precise answer?",
    "I do not have enough information to answer this question.",
    "I do not have enough information to determine this.",
}


# ── KG lightweight wrapper (for use inside this module) ───────────────────────

class _KG:
    """Minimal KG wrapper for dataset creation; loaded from pickle."""
    def __init__(self):
        self.G                = nx.DiGraph()
        self.kb_to_nodes      = defaultdict(list)
        self.kb_meta          = {}
        self.node_to_kbs      = defaultdict(set)
        self.variable_nodes   = set()
        self.reinforced_paths = {}


def load_kg_checkpoint(phase: str, path: str) -> Optional[_KG]:
    for fname in [f"kg_{phase}_clean.pkl", f"kg_{phase}.pkl"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            kg = _KG()
            kg.G                = data["graph"]
            kg.kb_to_nodes      = defaultdict(list, data["kb_to_nodes"])
            kg.kb_meta          = data["kb_meta"]
            kg.node_to_kbs      = defaultdict(set, data["node_to_kbs"])
            kg.variable_nodes   = data.get("variable_nodes", set())
            kg.reinforced_paths = data.get("reinforced_paths", {})
            print(f"✅ Loaded {phase} ← {fpath}  "
                  f"({kg.G.number_of_nodes()} nodes, "
                  f"{kg.G.number_of_edges()} edges)")
            return kg
    print(f"❌ No checkpoint for {phase} in {path}")
    return None


# ── Cache builder ─────────────────────────────────────────────────────────────

class KGCache:
    """
    Pre-encodes all variables and queries against KG node embeddings
    so that the main construction loop runs in O(1) per lookup.
    """

    def __init__(self, kg: _KG, sbert: SentenceTransformer,
                 all_samples: list[dict]):
        self.kg     = kg
        self.sbert  = sbert
        G_undir     = kg.G.to_undirected()
        all_nodes   = list(kg.G.nodes())

        print(f"  Encoding {len(all_nodes)} KG nodes …")
        node_matrix = sbert.encode(
            all_nodes, batch_size=512, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )

        # Collect unique strings
        all_vars    = set()
        all_queries = set()
        for s in all_samples:
            all_queries.add(s["query"])
            for v in s["state"].get("known_variables", []):
                if v: all_vars.add(v)
            for v in s["state"].get("missing_variables", []):
                if v: all_vars.add(v)

        var_list   = list(all_vars)
        query_list = list(all_queries)

        print(f"  Encoding {len(var_list)} variables …")
        var_embs = sbert.encode(var_list, batch_size=512,
                                show_progress_bar=True,
                                convert_to_numpy=True,
                                normalize_embeddings=True)

        print(f"  Encoding {len(query_list)} queries …")
        query_embs = sbert.encode(query_list, batch_size=512,
                                  show_progress_bar=True,
                                  convert_to_numpy=True,
                                  normalize_embeddings=True)

        # Specificity cache
        print("  Pre-scoring variable specificity …")
        generic_embs = sbert.encode(
            GENERIC_ANCHORS, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self._specificity: dict[str, bool] = {}
        if var_list:
            sims_to_generic = var_embs @ generic_embs.T
            for var, row in zip(var_list, sims_to_generic):
                self._specificity[var] = float(row.max()) < SPECIFICITY_THRESHOLD

        self._generic_embs = generic_embs

        # var → KG nodes
        print("  Mapping variables → KG nodes …")
        self._var_to_nodes: dict[str, list] = {}
        CHUNK = 1000
        for i in range(0, len(var_list), CHUNK):
            bv = var_list[i:i+CHUNK]
            be = var_embs[i:i+CHUNK]
            sims = be @ node_matrix.T
            for j, var in enumerate(bv):
                row = sims[j]
                top = row.argsort()[::-1][:5]
                self._var_to_nodes[var] = [
                    (all_nodes[k], float(row[k]))
                    for k in top if float(row[k]) >= NODE_MATCH_THRESH
                ]

        # query → KG nodes
        print("  Mapping queries → KG nodes …")
        self._query_to_nodes: dict[str, list] = {}
        for i in range(0, len(query_list), CHUNK):
            bq = query_list[i:i+CHUNK]
            be = query_embs[i:i+CHUNK]
            sims = be @ node_matrix.T
            for j, q in enumerate(bq):
                row = sims[j]
                top = row.argsort()[::-1][:5]
                self._query_to_nodes[q] = [
                    (all_nodes[k], float(row[k]))
                    for k in top if float(row[k]) >= NODE_MATCH_THRESH
                ]

        self._query_emb_map = {
            q: query_embs[i] for i, q in enumerate(query_list)
        }

        # Ego + triple caches
        all_seeds = set()
        for matches in (
            list(self._var_to_nodes.values())
            + list(self._query_to_nodes.values())
        ):
            for n, _ in matches:
                all_seeds.add(n)

        print(f"  Building ego caches for {len(all_seeds)} seed nodes …")
        self._ego: dict[str, set] = {}
        for node in tqdm(all_seeds, desc="  Ego graphs"):
            try:
                ego = nx.ego_graph(G_undir, node, radius=HOP_RADIUS)
                self._ego[node] = set(ego.nodes())
            except nx.NodeNotFound:
                self._ego[node] = {node}

        print("  Pre-computing triples per seed node …")
        self._triples: dict[str, list] = {}
        for sn in tqdm(all_seeds, desc="  Node triples"):
            ego_ns = self._ego.get(sn, {sn})
            subG   = kg.G.subgraph(ego_ns)
            edges  = sorted(
                subG.edges(data=True),
                key=lambda x: x[2].get("final_weight",
                                        x[2].get("weight", 0)),
                reverse=True,
            )
            triples, seen = [], set()
            for u, v, d in edges[:MAX_TRIPLES * 3]:
                rel  = d.get("relation", "related_to").replace("_", " ")
                line = f"{u} | {rel} | {v}"
                if line not in seen:
                    seen.add(line)
                    triples.append(line)
                if len(triples) >= MAX_TRIPLES * 2:
                    break
            self._triples[sn] = triples

        self._var_node_adj: dict[str, set] = {
            vn: (set(kg.G.predecessors(vn)) | set(kg.G.successors(vn)))
            for vn in kg.variable_nodes
        }
        print("✅ All caches built.\n")

    # ── lookup helpers ────────────────────────────────────────────────────────

    def is_specific(self, text: str) -> bool:
        if not text or len(text.strip()) < 4:
            return False
        if text in self._specificity:
            return self._specificity[text]
        emb = self.sbert.encode(
            [text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0]
        result = float((self._generic_embs @ emb).max()) < SPECIFICITY_THRESHOLD
        self._specificity[text] = result
        return result

    def var_nodes(self, var: str, top_k: int = 3) -> list:
        return self._var_to_nodes.get(var, [])[:top_k]

    def query_nodes(self, query: str, top_k: int = 5) -> list:
        return self._query_to_nodes.get(query, [])[:top_k]

    def filter_triples(self, query: str, triples: list) -> list:
        if not triples:
            return []
        q_emb = self._query_emb_map.get(query)
        if q_emb is None:
            return triples[:MAX_TRIPLES]
        t_embs = self.sbert.encode(
            triples, normalize_embeddings=True, convert_to_numpy=True
        )
        sims   = q_emb @ t_embs.T
        scored = sorted(zip(sims, triples), reverse=True)
        return [t for s, t in scored if float(s) >= TRIPLE_REL_THRESH][:MAX_TRIPLES]


# ── Core graph helpers ────────────────────────────────────────────────────────

def _anonymize_var_nodes(triples: list[str]) -> list[str]:
    counter: dict[str, str] = {}
    cleaned = []
    for t in triples:
        parts = t.split(" | ")
        if len(parts) != 3:
            cleaned.append(t)
            continue
        new = []
        for part in parts:
            if re.match(r"^\?var_", part):
                if part not in counter:
                    counter[part] = f"?unknown_{len(counter)+1}"
                new.append(counter[part])
            else:
                new.append(part)
        cleaned.append(" | ".join(new))
    return cleaned


def get_subgraph_triples(
    query: str,
    known_variables: list[str],
    missing_variables: list[str],
    action: str,
    cache: KGCache,
) -> tuple[list[str], list, list]:
    if cache is None:
        return [], [], []

    seed_nodes    = set()
    matched_nodes = []

    for node, score in cache.query_nodes(query, top_k=3):
        seed_nodes.add(node)
        matched_nodes.append(("query", node, score))

    for var in known_variables:
        for node, score in cache.var_nodes(var, top_k=2):
            seed_nodes.add(node)
            matched_nodes.append((var, node, score))

    if not seed_nodes:
        return [], [], []

    raw_triples: list[str] = []
    seen: set[str] = set()
    for sn in seed_nodes:
        for line in cache._triples.get(sn, []):
            if line not in seen:
                seen.add(line)
                raw_triples.append(line)

    var_triples: list[str] = []
    if action == "ASK":
        count = 0
        for vn, adj_set in cache._var_node_adj.items():
            if count >= MAX_VAR_SHOWN:
                break
            hit = adj_set & seed_nodes
            if hit:
                line = f"{next(iter(hit))} | requires | {vn}"
                if line not in seen:
                    seen.add(line)
                    var_triples.append(line)
                    count += 1

    all_triples = raw_triples + var_triples
    filtered    = cache.filter_triples(query, all_triples)
    anonymized  = _anonymize_var_nodes(filtered)
    return anonymized, matched_nodes, list(seed_nodes)


def cap_requires_edges(triples: list[str], max_requires: int = 3,
                       action: str = "ANSWER") -> list[str]:
    if action == "ANSWER":
        return [t for t in triples
                if not ("requires" in t and "?unknown" in t)]
    requires, others = [], []
    for t in triples:
        parts = t.split(" | ")
        if (len(parts) == 3
                and "requires" in parts[1]
                and "?unknown" in parts[2]):
            requires.append(t)
        else:
            others.append(t)
    return others + requires[:max_requires]


# ── Variable resolution helpers ───────────────────────────────────────────────

def infer_effective_missing(
    query: str,
    known_variables: list[str],
    missing_variables: list[str],
    graph_triples: list[str],
    action: str,
    cache: Optional[KGCache] = None,
    cumulative_known: Optional[list[str]] = None,
    still_missing_from_history: Optional[list[str]] = None,
) -> list[str]:
    def _finalize(results):
        specific = [r for r in results
                    if cache is None or cache.is_specific(r)]
        return specific if specific else []

    if missing_variables:
        resolved = set(cumulative_known or [])
        filtered = [m for m in missing_variables if m not in resolved]
        if filtered:
            return _finalize(filtered)

    if still_missing_from_history:
        filtered = [m for m in still_missing_from_history
                    if m not in set(cumulative_known or [])]
        if filtered:
            return _finalize(filtered)

    requires_subjects = []
    for t in graph_triples:
        parts = t.split(" | ")
        if (len(parts) == 3 and "requires" in parts[1]
                and parts[2].startswith("?unknown")
                and not parts[0].startswith("?")):
            subj = parts[0].strip()
            if subj not in requires_subjects:
                requires_subjects.append(subj)

    if requires_subjects and action in ("ASK", "ABSTAIN"):
        return _finalize([f"details about '{requires_subjects[0]}'"])

    q_lower = query.lower()
    if action == "ASK":
        if q_lower.startswith("what"):
            words = query.split()
            if len(words) > 1:
                return _finalize([f"'{words[1]}' information"])
        for w in ["who", "which", "where", "when", "how"]:
            if w in q_lower:
                return _finalize([f"'{w}' referent for this query"])
        return _finalize([f"context for: '{query[:40].rstrip('?').strip()}'"])

    if action == "ABSTAIN":
        specific = [k for k in (known_variables or [])
                    if cache is None or cache.is_specific(k)]
        topic = specific[0] if specific else (
            known_variables[0] if known_variables
            else query.split()[0]
        )
        return _finalize(
            [f"information about '{topic}' absent from knowledge base"]
        )
    return []


def accumulate_variables_from_history(
    samples_in_dialogue: list[dict], current_turn_id: int
) -> tuple[list, list, list]:
    cumulative_known:    set[str] = set()
    cumulative_resolved: set[str] = set()
    cumulative_missing:  set[str] = set()

    for s in samples_in_dialogue:
        if (s["metadata"].get("turn_id") or 0) >= current_turn_id:
            break
        for v in s["state"].get("known_variables", []):
            if v:
                cumulative_known.add(v)
        s_missing = s["state"].get("missing_variables", [])
        if s["action"] == "ASK" and s_missing:
            cumulative_resolved.add(s_missing[0])
            cumulative_known.add(s_missing[0])
        for v in s_missing:
            if v:
                cumulative_missing.add(v)

    still_missing = cumulative_missing - cumulative_resolved
    return (sorted(cumulative_known),
            sorted(cumulative_resolved),
            sorted(still_missing))


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_ask_question(
    query: str,
    effective_missing: list[str],
    known_variables: list[str],
    graph_triples: list[str],
    sbert: Optional[SentenceTransformer] = None,
) -> str:
    if not effective_missing:
        subject = (known_variables[0] if known_variables
                   else query.split()[0] if query else "this topic")
        return f"Could you provide more details about {subject}?"

    top_missing   = effective_missing[0]
    clean_missing = re.sub(
        r"^(details about|context for:|information about)\s*",
        "", top_missing, flags=re.IGNORECASE,
    ).strip("'\" ")

    anchor = None
    if sbert and known_variables and clean_missing:
        kv_embs  = sbert.encode(
            [clean_missing] + known_variables,
            normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        sims     = kv_embs[1:] @ kv_embs[0]
        best_idx = int(sims.argmax())
        if float(sims[best_idx]) >= 0.20:
            anchor = known_variables[best_idx]

    if not anchor:
        anchor = known_variables[0] if known_variables else None

    if not anchor:
        return f"Could you specify {clean_missing}?"
    if anchor.lower() == clean_missing.lower():
        return f"Could you provide more details about {anchor}?"
    return f"Regarding {anchor}: could you specify {clean_missing}?"


def _summarize_graph(graph_triples: list[str],
                     matched_nodes: list, action: str) -> str:
    if not graph_triples:
        return ("The graph context contains no relevant connections "
                "for the entities in this query.")

    entities, relations = [], []
    for t in graph_triples[:6]:
        parts = t.split(" | ")
        if len(parts) == 3:
            subj, rel, obj = parts
            if not subj.startswith("?") and len(subj) > 2:
                entities.append(subj.strip())
            if not obj.startswith("?") and len(obj) > 2:
                entities.append(obj.strip())
            if rel.strip() not in {"requires", "related to"} and len(rel) > 3:
                relations.append(rel.strip())

    entities  = list(dict.fromkeys(entities))[:4]
    relations = list(dict.fromkeys(relations))[:3]

    match_summary = ""
    top_matches   = [(var, node)
                     for var, node, sc in matched_nodes
                     if sc >= NODE_MATCH_THRESH][:3]
    if top_matches:
        parts = [f"'{node}' (matched from '{var}')"
                 for var, node in top_matches]
        match_summary = f"Query terms matched KG nodes: {'; '.join(parts)}. "

    entity_str   = ", ".join(entities)   if entities   else "the query topic"
    relation_str = ", ".join(relations)  if relations  else "various relations"

    if action == "ANSWER":
        return (f"{match_summary}"
                f"Graph traversal found connected nodes involving: {entity_str}. "
                f"Key relations present: {relation_str}. "
                f"The path from known entities to an answer is complete.")
    if action == "ASK":
        req_note = (
            " Variable placeholder nodes (requires edges) indicate "
            "missing information in the graph."
            if any("requires" in t for t in graph_triples) else ""
        )
        return (f"{match_summary}"
                f"Graph traversal found partial connections involving: "
                f"{entity_str}. Relations seen: {relation_str}.{req_note} "
                f"The path cannot be completed without additional information.")
    return (f"{match_summary}"
            f"Graph traversal found connections involving: {entity_str}, "
            f"but these do not connect to information that resolves the query. "
            f"The missing information is not obtainable through clarification.")


def _build_user_turn(
    sample: dict,
    graph_triples: list[str],
    history=None,
    cumulative_resolved: Optional[set] = None,
    cumulative_known: Optional[list] = None,
    all_known: Optional[list] = None,
) -> tuple[str, list]:
    known   = all_known or sample["state"].get("known_variables", [])
    missing = sample["state"].get("missing_variables", [])

    effective_missing = [m for m in missing
                         if m not in (cumulative_resolved or set())]

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

    state_block = ""
    if history:
        resolved     = cumulative_resolved or set()
        still_miss   = [m for m in effective_missing if m not in resolved]
        if still_miss:
            state_block += (
                "<remaining_unknowns>\n"
                + "\n".join(f"- {m}" for m in still_miss)
                + "\n</remaining_unknowns>\n"
            )
        if resolved:
            state_block += (
                "<resolved_variables>\n"
                + "\n".join(f"- {r}" for r in sorted(resolved))
                + "\n</resolved_variables>\n"
            )
        if state_block:
            state_block += "\n"

    graph_block = (
        "<graph_context>\n"
        + "\n".join(graph_triples)
        + "\n</graph_context>"
    ) if graph_triples else (
        "<graph_context>\nNo relevant nodes found in knowledge graph.\n</graph_context>"
    )

    missing_block = (
        "<missing_variables>\n"
        + "\n".join(f"- {m}" for m in effective_missing)
        + "\n</missing_variables>"
    ) if effective_missing else "<missing_variables>\nnone\n</missing_variables>"

    user_turn = (
        f"{history_block}{state_block}"
        f"<query>\n{sample['query']}\n</query>\n\n"
        f"<known_variables>\n"
        f"{', '.join(known) if known else 'none identified'}\n"
        f"</known_variables>\n\n"
        f"{graph_block}\n\n{missing_block}\n\n"
        f"Search the graph context for relevant nodes and decide the correct action."
    )
    return user_turn, effective_missing


def _build_assistant_turn(
    sample: dict,
    graph_triples: list[str],
    matched_nodes: list,
    effective_missing: list[str],
    cumulative_known: Optional[list] = None,
    sbert: Optional[SentenceTransformer] = None,
) -> str:
    action    = sample["action"]
    known     = sample["state"].get("known_variables", [])
    fm        = sample["state"].get("failure_mode", "COMPLETE")
    all_known = list(set(known + (cumulative_known or [])))

    query_subject = (", ".join(all_known[:3])
                     if all_known else f"'{sample['query'][:50]}'")
    graph_summary = _summarize_graph(graph_triples, matched_nodes, action)

    if action == "ANSWER":
        var_check = (
            f"Known variables ({', '.join(all_known[:3]) if all_known else 'query entities'}) "
            f"are present in the graph. No critical variables are missing. "
            f"Failure mode: {fm} (complete information state)."
        )
        decision_rationale = (
            "The graph provides a complete reasoning path "
            "from the query entities to a resolvable answer."
        )
        justification = (
            "Graph traversal is complete — all required nodes "
            "are connected and no missing variables block the answer."
        )
        response_line = ""

    elif action == "ASK":
        missing_str = (
            ", ".join(f"'{m}'" for m in effective_missing[:2])
            if effective_missing else "'further context'"
        )
        var_check = (
            f"Known: {', '.join(all_known[:3]) if all_known else 'query entities only'}. "
            f"Required but absent from graph: {missing_str}. Failure mode: {fm}."
        )
        decision_rationale = (
            f"The graph has partial connections for this topic but cannot "
            f"complete the reasoning path without: {missing_str}."
        )
        ask_q = _build_ask_question(
            sample["query"], effective_missing, all_known,
            graph_triples, sbert,
        )
        justification = ask_q
        response_line = f"\n\n<clarification_question>\n{ask_q}\n</clarification_question>"

    else:  # ABSTAIN
        missing_str = (
            ", ".join(f"'{m}'" for m in effective_missing[:2])
            if effective_missing else "'required context'"
        )
        var_check = (
            f"Known: {', '.join(all_known[:3]) if all_known else 'none'}. "
            f"Missing: {missing_str}. Failure mode: {fm} — "
            f"information is absent from the graph entirely."
        )
        decision_rationale = (
            f"The graph lacks nodes to resolve this query. "
            f"Missing: {missing_str}. No clarification from the user can fill this gap."
        )
        justification = (
            f"Graph has no resolvable path — "
            f"{missing_str} is entirely absent from the knowledge base."
        )
        response_line = ""

    return (
        f"<reasoning>\n"
        f"Step 1 — Query subject: {query_subject}. "
        f"Query asks: '{sample['query'][:80]}'\n"
        f"Step 2 — Graph search: {graph_summary}\n"
        f"Step 3 — Variable check: {var_check}\n"
        f"Step 4 — Decision rationale: {decision_rationale}\n"
        f"</reasoning>\n\n"
        f"<decision>\n{action}\n</decision>\n\n"
        f"<justification>\n{justification}\n</justification>"
        f"{response_line}"
    )


def _build_conversation_history(
    samples_in_dialogue: list[dict], current_turn_id: int
) -> tuple[Optional[list], set]:
    history: list[dict] = []
    resolved: set[str] = set()

    for s in samples_in_dialogue:
        t_id = s["metadata"].get("turn_id") or 0
        if t_id >= current_turn_id:
            break
        s_action  = s["action"]
        s_missing = s["state"].get("missing_variables", [])
        s_response = s["response"]

        # clean dummy responses
        if s_response.strip() in DUMMY_RESPONSES:
            if s_action == "ASK" and s_missing:
                s_response = (f"[Clarification requested: "
                              f"'{s_missing[0]}' is needed to proceed]")
            elif s_action == "ABSTAIN":
                s_response = "[Cannot answer: insufficient context]"

        resolved_var = None
        if s_action == "ASK" and s_missing:
            resolved_var = s_missing[0]
            resolved.add(resolved_var)

        history.append({
            "turn_id"          : t_id,
            "query"            : s["query"],
            "action"           : s_action,
            "response"         : s_response[:120],
            "missing_variables": s_missing,
            "resolved_variable": resolved_var,
        })

    return (history or None), resolved


# ── Quality filter ────────────────────────────────────────────────────────────

def _is_valid(known: list, graph_triples: list, action: str,
              effective_missing: list) -> tuple[bool, str]:
    if not known and not graph_triples and action == "ASK":
        return False, "zero_signal_ask"
    if not graph_triples and action == "ANSWER" and not known:
        return False, "no_graph_answer"
    if (action == "ASK" and not effective_missing
            and not any("requires" in t for t in graph_triples)):
        return False, "ask_no_missing_no_var_nodes"
    if action == "ANSWER" and graph_triples and known:
        known_lower = {k.lower().strip() for k in known if k}
        has_match   = any(
            any(k in t.lower() for k in known_lower)
            for t in graph_triples
        )
        if not has_match:
            return False, "answer_graph_irrelevant"
    if action == "ANSWER" and not graph_triples:
        return False, "answer_empty_graph"
    return True, "ok"


# ── Main constructor ──────────────────────────────────────────────────────────

def build_ft_dataset(
    all_samples:   list[dict],
    kg_ckpt_path:  str,
    output_dir:    str,
    sbert:         Optional[SentenceTransformer] = None,
) -> dict[str, list]:
    """
    Build train/val/test fine-tuning splits from the unified dataset and KG.

    Args:
        all_samples:  List of samples from unified_train.json (populated).
        kg_ckpt_path: Directory containing kg_G2.pkl (or kg_G1.pkl).
        output_dir:   Directory to write ft_train/val/test_{debug,mistral}.jsonl.
        sbert:        Sentence transformer (loaded if None).

    Returns:
        Dict with keys "train", "val", "test" each containing a list of samples.
    """
    os.makedirs(output_dir, exist_ok=True)

    if sbert is None:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

    kg = load_kg_checkpoint("G2", kg_ckpt_path) \
         or load_kg_checkpoint("G1", kg_ckpt_path)

    if kg is None:
        raise FileNotFoundError(
            f"No KG checkpoint found in {kg_ckpt_path}. "
            "Run knowledge_graph.py first."
        )

    cache = KGCache(kg, sbert, all_samples)

    # ── Separate single-turn and multi-turn ───────────────────
    dialogues:   dict[str, list] = defaultdict(list)
    single_turn: list[dict]      = []

    for s in all_samples:
        if (s["metadata"].get("multi_turn")
                and s["metadata"].get("dialogue_id")):
            dialogues[s["metadata"]["dialogue_id"]].append(s)
        else:
            single_turn.append(s)

    for dlg_id in dialogues:
        dialogues[dlg_id].sort(
            key=lambda x: x["metadata"].get("turn_id") or 0
        )

    print(f"\nMulti-turn dialogues : {len(dialogues)}")
    print(f"Single-turn samples  : {len(single_turn)}")

    ft_samples: list[dict] = []
    skipped     = Counter()
    ft_id_ctr   = 0

    # ── Single-turn loop ──────────────────────────────────────
    print("\nProcessing single-turn …")
    for s in tqdm(single_turn, desc="Single-turn"):
        known   = s["state"].get("known_variables", [])
        missing = s["state"].get("missing_variables", [])
        action  = s["action"]

        graph_triples, matched_nodes, seed_nodes = get_subgraph_triples(
            s["query"], known, missing, action, cache
        )
        graph_triples     = cap_requires_edges(graph_triples, action=action)
        effective_missing = infer_effective_missing(
            s["query"], known, missing, graph_triples, action, cache
        )

        valid, reason = _is_valid(known, graph_triples, action, effective_missing)
        if not valid:
            skipped[reason] += 1
            continue

        user_turn, _ = _build_user_turn(s, graph_triples)
        asst_turn    = _build_assistant_turn(
            s, graph_triples, matched_nodes, effective_missing, sbert=sbert
        )
        sample = _make_sample(s, graph_triples, matched_nodes, seed_nodes,
                              effective_missing, None, set(),
                              f"ft_{ft_id_ctr:07d}")
        ft_samples.append(sample)
        ft_id_ctr += 1

    # ── Multi-turn loop ───────────────────────────────────────
    print("\nProcessing multi-turn dialogues …")
    for dlg_id, turns in tqdm(dialogues.items(), desc="Dialogues"):
        for turn in turns:
            t_id    = turn["metadata"].get("turn_id") or 0
            known   = turn["state"].get("known_variables", [])
            missing = turn["state"].get("missing_variables", [])
            action  = turn["action"]

            cum_known, cum_resolved_list, still_miss = \
                accumulate_variables_from_history(turns, t_id)
            cum_resolved      = set(cum_resolved_list)
            all_known_turn    = list(set(known + cum_known))

            history, _ = _build_conversation_history(turns, t_id)

            graph_triples, matched_nodes, seed_nodes = get_subgraph_triples(
                turn["query"], all_known_turn, missing, action, cache
            )
            graph_triples     = cap_requires_edges(graph_triples, action=action)
            effective_missing = infer_effective_missing(
                turn["query"], all_known_turn, missing,
                graph_triples, action, cache,
                cumulative_known=all_known_turn,
                still_missing_from_history=still_miss,
            )
            effective_missing = [m for m in effective_missing
                                 if m not in cum_resolved]

            valid, reason = _is_valid(
                all_known_turn, graph_triples, action, effective_missing
            )
            if not valid:
                skipped[reason] += 1
                continue

            sample = _make_sample(
                turn, graph_triples, matched_nodes, seed_nodes,
                effective_missing, history, cum_resolved,
                f"ft_{ft_id_ctr:07d}",
                cumulative_known=cum_known,
                still_missing_from_history=still_miss,
                sbert=sbert,
            )
            ft_samples.append(sample)
            ft_id_ctr += 1

    print(f"\n✅ Built {len(ft_samples)} samples  (skipped: {dict(skipped)})")

    # ── Train/val/test split ──────────────────────────────────
    all_dlg_ids   = list({s["dialogue_id"] for s in ft_samples if s["dialogue_id"]})
    all_single_ids = [s["ft_id"] for s in ft_samples if not s["dialogue_id"]]
    random.shuffle(all_dlg_ids)
    random.shuffle(all_single_ids)

    n_d = len(all_dlg_ids)
    train_dlgs = set(all_dlg_ids[:int(0.70 * n_d)])
    val_dlgs   = set(all_dlg_ids[int(0.70 * n_d):int(0.85 * n_d)])

    n_s = len(all_single_ids)
    train_singles = set(all_single_ids[:int(0.70 * n_s)])
    val_singles   = set(all_single_ids[int(0.70 * n_s):int(0.85 * n_s)])

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for s in ft_samples:
        if s["dialogue_id"]:
            key = ("train" if s["dialogue_id"] in train_dlgs
                   else "val" if s["dialogue_id"] in val_dlgs
                   else "test")
        else:
            key = ("train" if s["ft_id"] in train_singles
                   else "val" if s["ft_id"] in val_singles
                   else "test")
        splits[key].append(s)

    for name, split in splits.items():
        split.sort(key=lambda x: (x["dialogue_id"] or x["ft_id"],
                                   x["turn_id"] or 0))
        # debug JSON
        p = os.path.join(output_dir, f"ft_{name}_debug.json")
        with open(p, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  Saved {name} debug   → {p}")

        # clean JSONL
        p = os.path.join(output_dir, f"ft_{name}.jsonl")
        with open(p, "w") as f:
            for s in split:
                s_clean = {k: v for k, v in s.items() if k != "_debug"}
                f.write(json.dumps(s_clean) + "\n")

        # Mistral-format JSONL
        p = os.path.join(output_dir, f"ft_{name}_mistral.jsonl")
        with open(p, "w") as f:
            for s in split:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")
        print(f"  Saved {name} Mistral → {p}")

    for name, split in splits.items():
        acts = Counter(s["action"] for s in split)
        print(f"  {name:5}: {len(split):6}  "
              + "  ".join(f"{a}:{acts[a]}" for a in ["ANSWER","ASK","ABSTAIN"]))

    return splits


def _make_sample(
    s: dict,
    graph_triples: list,
    matched_nodes: list,
    seed_nodes: list,
    effective_missing: list,
    history,
    cum_resolved: set,
    ft_id: str,
    cumulative_known: Optional[list] = None,
    still_missing_from_history: Optional[list] = None,
    sbert: Optional[SentenceTransformer] = None,
) -> dict:
    all_known = list(set(
        s["state"].get("known_variables", []) + (cumulative_known or [])
    ))
    user_turn, _ = _build_user_turn(
        s, graph_triples, history, cum_resolved,
        cumulative_known=cumulative_known, all_known=all_known,
    )
    asst_turn = _build_assistant_turn(
        s, graph_triples, matched_nodes, effective_missing,
        cumulative_known=all_known, sbert=sbert,
    )
    return {
        "ft_id"            : ft_id,
        "source_id"        : s["id"],
        "source"           : s["metadata"]["source"],
        "action"           : s["action"],
        "difficulty"       : s["state"].get("difficulty", "medium"),
        "failure_mode"     : s["state"].get("failure_mode", ""),
        "multi_turn"       : s["metadata"].get("multi_turn", False),
        "turn_id"          : s["metadata"].get("turn_id"),
        "dialogue_id"      : s["metadata"].get("dialogue_id"),
        "num_known"        : len(all_known),
        "num_missing"      : len(effective_missing),
        "num_triples"      : len(graph_triples),
        "num_matched_nodes": len(seed_nodes),
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_turn},
            {"role": "assistant", "content": asst_turn},
        ],
        "_debug": {
            "query"                     : s["query"],
            "known_variables"           : s["state"].get("known_variables", []),
            "cumulative_known"          : cumulative_known or [],
            "all_known_this_turn"       : all_known,
            "missing_variables"         : s["state"].get("missing_variables", []),
            "still_missing_from_history": still_missing_from_history or [],
            "effective_missing"         : effective_missing,
            "graph_triples"             : graph_triples,
            "matched_nodes"             : [(v, n, round(sc, 3))
                                           for v, n, sc in matched_nodes[:5]],
            "seed_nodes"                : list(seed_nodes)[:10],
            "original_response"         : s["response"],
            "history_length"            : len(history) if history else 0,
            "resolved_vars"             : list(cum_resolved),
            "context_preview"           : (
                s["context"]["documents"][0]["text"][:200]
                if s["context"]["documents"] else ""
            ),
        },
    }