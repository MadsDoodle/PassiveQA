"""
Three-phase Knowledge Graph construction for PassiveQA.

Phase 1 (G0): Named entity extraction + co-occurrence edges from KB texts.
Phase 2 (G1): Semantic validation — prune edges with low embedding similarity.
Phase 3 (G2): Query-guided decision reinforcement — reinforce ANSWER paths,
              penalise ABSTAIN paths, inject ?var placeholder nodes for
              recoverable missing variables.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

STOPWORDS = {
    "he","she","it","they","his","her","its","their","him","them",
    "this","that","these","those","i","we","you","who","which","what",
    "where","when","how","by","the","a","an","was","were","been",
    "being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","need","dare","ought",
}

NAMED_ENT_TYPES = {
    "PERSON","ORG","GPE","LOC","WORK_OF_ART",
    "EVENT","LAW","PRODUCT","NORP","FAC","LANGUAGE",
}

SPACY_TO_CAT = {
    "PERSON":      "Person",
    "ORG":         "Organization",
    "GPE":         "Location",
    "LOC":         "Location",
    "WORK_OF_ART": "Work",
    "EVENT":       "Event",
    "LAW":         "Concept",
    "PRODUCT":     "Work",
    "NORP":        "Organization",
    "FAC":         "Location",
    "LANGUAGE":    "Concept",
}

GENERIC_PHRASES = {
    "the season","the year","the time","the game","the team",
    "the show","the film","one","two","three","the series",
    "the episode","the album","the song","the book",
}

CFG = {
    "sem_threshold"    : 0.50,
    "min_edge_weight"  : 0.10,
    "max_var_hops"     : 3,
    "reinforce_delta"  : 0.20,
    "penalise_delta"   : 0.15,
}


# ── KG class ──────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Wrapper around a NetworkX DiGraph with PassiveQA-specific metadata.
    """

    def __init__(self) -> None:
        self.G                = nx.DiGraph()
        self.kb_to_nodes      = defaultdict(list)
        self.kb_to_triples    = defaultdict(list)
        self.kb_meta          : dict[str, Any] = {}
        self.node_to_kbs      = defaultdict(set)
        self.variable_nodes   : set[str] = set()
        self.reinforced_paths : dict[str, Any] = {}

    # ── Basic graph ops ───────────────────────────────────────────────────────

    def add_node(self, node_id: str, **attrs) -> None:
        if not self.G.has_node(node_id):
            self.G.add_node(node_id, **attrs)

    def add_edge(self, src: str, dst: str, **attrs) -> None:
        if not self.G.has_edge(src, dst):
            self.G.add_edge(src, dst, **attrs)
        else:
            # Accumulate weight
            w = self.G[src][dst].get("weight", 1.0)
            self.G[src][dst]["weight"] = w + attrs.get("weight", 1.0)

    def record_triple(self, kb_id: str,
                      subj: str, rel: str, obj: str) -> None:
        triple = (subj, rel, obj)
        if triple not in self.kb_to_triples[kb_id]:
            self.kb_to_triples[kb_id].append(triple)


# ── Helper: normalise entity text ─────────────────────────────────────────────

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_valid_entity(text: str) -> bool:
    n = normalise(text)
    return (len(n) >= 3 and
            n not in STOPWORDS and
            n not in GENERIC_PHRASES and
            not n.isdigit())


# ── Phase 1: KB → entity extraction + co-occurrence edges ────────────────────

def build_phase1(kb_entries: list[dict],
                 nlp=None) -> KnowledgeGraph:
    """
    Build G0: extract named entities from KB texts and link co-occurring
    entities within the same chunk.

    Args:
        kb_entries: List of KB entry dicts (from baseline_rag.build_kb).
        nlp:        Loaded spaCy model (en_core_web_sm). Loaded if None.

    Returns:
        KnowledgeGraph instance (G0).
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["senter"])

    kg = KnowledgeGraph()

    for entry in tqdm(kb_entries, desc="Phase 1 — entity extraction"):
        kb_id  = entry["kb_id"]
        source = entry.get("source", "unknown")

        kg.kb_meta[kb_id] = {
            "source"           : source,
            "action_dist"      : entry.get("action_distribution", {}),
            "linked_queries"   : entry.get("linked_queries", []),
            "linked_actions"   : entry.get("linked_actions", []),
            "linked_sample_ids": entry.get("linked_sample_ids", []),
        }

        text = entry.get("text", "")
        if not text.strip():
            continue

        doc   = nlp(text[:5000])   # cap at 5K chars for speed
        ents  = [e for e in doc.ents if e.label_ in NAMED_ENT_TYPES]
        nodes = []

        for ent in ents:
            if not _is_valid_entity(ent.text):
                continue
            node_id  = normalise(ent.text)
            category = SPACY_TO_CAT.get(ent.label_, "Concept")
            kg.add_node(node_id, label=node_id, category=category,
                        ent_type=ent.label_, sources={source})
            kg.kb_to_nodes[kb_id].append(node_id)
            kg.node_to_kbs[node_id].add(kb_id)
            nodes.append(node_id)

        # Co-occurrence edges within the chunk
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                kg.add_edge(nodes[i], nodes[j],
                            rel="co_occurs", weight=1.0,
                            source=source, kb_id=kb_id)
                kg.add_edge(nodes[j], nodes[i],
                            rel="co_occurs", weight=1.0,
                            source=source, kb_id=kb_id)
                kg.record_triple(kb_id, nodes[i], "co_occurs", nodes[j])

    print(f"\nG0: {kg.G.number_of_nodes()} nodes | "
          f"{kg.G.number_of_edges()} edges")
    return kg


# ── Phase 2: Semantic validation ──────────────────────────────────────────────

def phase2_semantic_validation(
    kg: KnowledgeGraph,
    kb_entries: list[dict],
    sbert: Optional[SentenceTransformer] = None,
    sem_threshold: float = CFG["sem_threshold"],
    batch_size: int = 512,
) -> KnowledgeGraph:
    """
    Prune edges where the two KB texts they span have low semantic similarity.

    Also adds cross-KB semantic edges where similarity exceeds the threshold.

    Returns:
        Updated KnowledgeGraph (G1).
    """
    if sbert is None:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("Phase 2 — semantic validation …")
    kb_text_map  = {e["kb_id"]: e["text"] for e in kb_entries}
    all_kb_ids   = list(kb_text_map.keys())

    print(f"  Encoding {len(all_kb_ids)} KB texts …")
    kb_matrix = sbert.encode(
        [kb_text_map[k] for k in all_kb_ids],
        batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    kb_id_to_idx = {kid: i for i, kid in enumerate(all_kb_ids)}

    # For each node, compute average KB-text similarity to decide edge validity
    edges_to_remove = []
    for src, dst, data in tqdm(kg.G.edges(data=True),
                                desc="  Pruning weak edges",
                                total=kg.G.number_of_edges()):
        src_kbs = list(kg.node_to_kbs.get(src, set()))
        dst_kbs = list(kg.node_to_kbs.get(dst, set()))
        if not src_kbs or not dst_kbs:
            edges_to_remove.append((src, dst))
            continue

        sims = []
        for sk in src_kbs[:5]:    # cap for speed
            for dk in dst_kbs[:5]:
                si = kb_id_to_idx.get(sk)
                di = kb_id_to_idx.get(dk)
                if si is not None and di is not None:
                    sims.append(float(np.dot(kb_matrix[si], kb_matrix[di])))

        if not sims or max(sims) < sem_threshold:
            edges_to_remove.append((src, dst))

    kg.G.remove_edges_from(edges_to_remove)
    print(f"  Pruned {len(edges_to_remove)} edges")

    # Remove isolated nodes
    isolated = [n for n in kg.G.nodes() if kg.G.degree(n) == 0]
    kg.G.remove_nodes_from(isolated)
    print(f"  Removed {len(isolated)} isolated nodes")
    print(f"G1: {kg.G.number_of_nodes()} nodes | "
          f"{kg.G.number_of_edges()} edges")
    return kg


# ── Phase 3: Query-guided decision reinforcement ──────────────────────────────

def phase3_query_refinement(
    kg: KnowledgeGraph,
    kb_entries: list[dict],
    nlp=None,
    reinforce_delta: float = CFG["reinforce_delta"],
    penalise_delta:  float = CFG["penalise_delta"],
    max_hops:        int   = CFG["max_var_hops"],
) -> KnowledgeGraph:
    """
    Phase 3 — Query-guided reinforcement on edge weights + ?var injection.

    For each linked query in each KB entry:
    - ANSWER queries: reinforce shortest paths → increase edge weights
    - ABSTAIN queries: penalise edges → decrease edge weights
    - ASK queries: inject ?var placeholder node for missing variable

    Returns:
        Updated KnowledgeGraph (G2).
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["senter"])

    print("Phase 3 — query-guided refinement …")
    G_undir = kg.G.to_undirected()

    def fast_path(src: str, tgt: str) -> Optional[list[str]]:
        try:
            p = nx.shortest_path(G_undir, src, tgt)
            return p if len(p) - 1 <= max_hops else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def inject_variable(anchor: str, query: str, sid: str) -> None:
        if not kg.G.has_node(anchor):
            return
        var_id = f"?var_{sid}"
        kg.add_node(var_id, label=var_id, category="Variable",
                    query=query, anchor=anchor)
        kg.add_edge(anchor, var_id, rel="has_missing_var", weight=1.0)
        kg.variable_nodes.add(var_id)

    def find_missing_entity(query: str) -> Optional[str]:
        q_doc  = nlp(query)
        q_ents = [normalise(e.text) for e in q_doc.ents
                  if e.label_ in NAMED_ENT_TYPES]
        missing = [e for e in q_ents if not kg.G.has_node(e)
                   and e not in STOPWORDS
                   and e not in GENERIC_PHRASES
                   and len(e) > 3]
        return missing[0] if missing else None

    for entry in tqdm(kb_entries, desc="  Phase 3"):
        kb_id  = entry["kb_id"]
        nodes  = kg.kb_to_nodes.get(kb_id, [])
        if not nodes:
            continue

        queries = entry.get("linked_queries", [])
        actions = entry.get("linked_actions", [])
        sids    = entry.get("linked_sample_ids", [])

        for query, action, sid in zip(queries, actions, sids):
            q_doc  = nlp(query)
            q_ents = [normalise(e.text) for e in q_doc.ents
                      if e.label_ in NAMED_ENT_TYPES
                      and _is_valid_entity(e.text)]

            # Reinforce ANSWER paths
            if action == "ANSWER":
                for src in q_ents:
                    for dst in nodes:
                        if src == dst:
                            continue
                        path = fast_path(src, dst)
                        if path:
                            for u, v in zip(path[:-1], path[1:]):
                                if kg.G.has_edge(u, v):
                                    kg.G[u][v]["weight"] = min(
                                        5.0,
                                        kg.G[u][v].get("weight", 1.0)
                                        + reinforce_delta)
                            key = f"{src}→{dst}"
                            kg.reinforced_paths[key] = {
                                "path": path, "action": "ANSWER",
                                "kb_id": kb_id, "query": query,
                            }

            # Penalise ABSTAIN paths
            elif action == "ABSTAIN":
                for src in q_ents:
                    for dst in nodes:
                        if src == dst:
                            continue
                        path = fast_path(src, dst)
                        if path:
                            for u, v in zip(path[:-1], path[1:]):
                                if kg.G.has_edge(u, v):
                                    kg.G[u][v]["weight"] = max(
                                        0.01,
                                        kg.G[u][v].get("weight", 1.0)
                                        - penalise_delta)

            # Inject ?var for ASK (recoverable missing variable)
            elif action == "ASK":
                missing_ent = find_missing_entity(query)
                if missing_ent:
                    anchor = nodes[0]
                    inject_variable(anchor, query, sid)

    print(f"G2: {kg.G.number_of_nodes()} nodes | "
          f"{kg.G.number_of_edges()} edges | "
          f"{len(kg.variable_nodes)} ?var nodes")
    return kg


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess_kg(kg: KnowledgeGraph,
                   kb_entries: list[dict]) -> KnowledgeGraph:
    """
    Clean G2 in-place without rerunning any phase.

    - Removes noise nodes (stopwords, pure numbers, single chars)
    - Prunes edges below minimum weight threshold
    - Removes nodes that lost all edges (except ?var nodes)
    """
    G = kg.G
    print("=" * 60)
    print("  KG POST-PROCESSING")
    print("=" * 60)

    # Step 1: Remove noise nodes
    def is_noise_node(node: str) -> bool:
        if node.startswith("?var_"):
            return False
        n = node.strip()
        return (len(n) <= 2 or n in STOPWORDS or n.isdigit() or
                n in GENERIC_PHRASES)

    noise_nodes = [n for n in G.nodes() if is_noise_node(n)]
    G.remove_nodes_from(noise_nodes)
    print(f"  Noise nodes removed   : {len(noise_nodes)}")

    # Step 2: Prune weak edges
    min_w = CFG["min_edge_weight"]
    weak_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d.get("weight", 1.0) < min_w]
    G.remove_edges_from(weak_edges)
    print(f"  Weak edges pruned     : {len(weak_edges)}")

    # Step 3: Remove newly isolated nodes (keep ?var nodes)
    isolated = [n for n in G.nodes()
                if G.degree(n) == 0 and not n.startswith("?var_")]
    G.remove_nodes_from(isolated)
    print(f"  Isolated nodes removed: {len(isolated)}")

    print(f"\n  Final G2: {G.number_of_nodes()} nodes | "
          f"{G.number_of_edges()} edges | "
          f"{len(kg.variable_nodes)} ?var nodes")
    return kg


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(kg: KnowledgeGraph, phase: str,
                    path: str = "./checkpoints/") -> None:
    """Pickle KG state to disk."""
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"kg_{phase}.pkl")
    data = {
        "graph"           : kg.G,
        "kb_to_nodes"     : dict(kg.kb_to_nodes),
        "kb_to_triples"   : dict(kg.kb_to_triples),
        "kb_meta"         : kg.kb_meta,
        "node_to_kbs"     : {k: v for k, v in kg.node_to_kbs.items()},
        "variable_nodes"  : kg.variable_nodes,
        "reinforced_paths": kg.reinforced_paths,
    }
    with open(fpath, "wb") as f:
        pickle.dump(data, f, protocol=4)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"Saved {phase} → {fpath}  ({size_mb:.1f} MB)")


def load_checkpoint(phase: str,
                    path: str = "./checkpoints/") -> KnowledgeGraph:
    """Load KG state from disk."""
    fpath = os.path.join(path, f"kg_{phase}.pkl")
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    kg = KnowledgeGraph()
    kg.G               = data["graph"]
    kg.kb_to_nodes     = defaultdict(list, data["kb_to_nodes"])
    kg.kb_to_triples   = defaultdict(list, data.get("kb_to_triples", {}))
    kg.kb_meta         = data["kb_meta"]
    kg.node_to_kbs     = defaultdict(set, data["node_to_kbs"])
    kg.variable_nodes  = data.get("variable_nodes", set())
    kg.reinforced_paths = data.get("reinforced_paths", {})
    print(f"Loaded {phase}: {kg.G.number_of_nodes()} nodes | "
          f"{kg.G.number_of_edges()} edges")
    return kg