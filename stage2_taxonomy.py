#!/usr/bin/env python3
"""Stage 2: resumable, weighted top-down taxonomy construction with semantic audit."""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

from llm_pool import (
    JsonlWriter, LLMPool, atomic_write_json, extract_json, file_hash, iter_jsonl, stable_hash,
    validate_schema_instance,
)
from pipeline_config import (
    ALLOW_MIXED_MODELS, AUTOPSY_DIR, EXTRA_REQUEST_BODY, GLOBAL_CONCURRENCY,
    HTTP_REQUEST_TIMEOUT, JUDGE_MODEL_ID, JUDGE_SERVERS, MAX_HTTP_RETRIES, PIPELINE_VERSION,
    REQUIRE_DISTINCT_JUDGE_MODEL, ROOT, SERVED_MODEL_ID, SERVERS, STRUCTURED_OUTPUT_MODE,
    TAXONOMY_DIR as OUTPUT_DIR,
)

CONCEPTS_FILE = AUTOPSY_DIR / "concepts_flat.jsonl"
ROOT_TITLE = "Code reasoning skills"
ROOT_DESCRIPTION = "Transferable algorithmic, semantic, contract, and failure-analysis skills used in programming tasks."
SUMMARY_LEVELS = 5
SUMMARY_BATCH_SIZE = 20
LIMITS = [2 ** (i + 1) for i in range(SUMMARY_LEVELS)]
MAX_BRANCHING = 15
MIN_BRANCHING = 5
MAX_TERMINAL_SIZE = 40
PROPOSE_MAX_ITEMS = 120
ASSIGN_BATCH = 100
SUMMARY_TOKEN_BUDGET = 40_000
MIN_UNIQUE_FOR_LEVEL = 12
MAX_CLUSTER_SHARE = 0.60
BALANCE_RETRIES = 3
CLUSTER_TEMPERATURE = 0.2
SEED = 11
LEAF_LIMIT: Optional[int] = None
AUDIT_SAMPLE_SIZE = 500
MIN_TAXONOMY_AUDIT_MEAN = 3.5
MAX_LOW_FIT_SHARE = 0.10

SUM_SYSTEM = """Compress each programming-reasoning skill into five nested summaries. Limits are 2, 4, 8, 16, and 32 words. Each level must refine the previous level. Describe the transferable mechanic, invariant, or pitfall—not the original task. Do not use identifiers or literal values. Return {"items":[{"id":integer,"summaries":[five strings]}]} for every input id."""
SUM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"id": {"type": "integer"}, "summaries": {
            "type": "array", "minItems": 5, "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 300}}},
        "required": ["id", "summaries"]}}}, "required": ["items"],
}

PROPOSE_SYSTEM = """Design mutually exclusive subcategories under this taxonomy node.
Ancestor path: {path}
Parent: {parent}
Parent definition: {description}

Each input is shown with the number of original concepts it represents. Bracketed `leaf-...` tokens only disambiguate duplicate descriptions and have no semantic meaning; never copy them into titles or descriptions. Propose {lo}-{hi} collectively exhaustive, roughly leaf-balanced categories along one consistent dimension. Titles must be 2-6 words, specific, and non-overlapping. Never use misc, other, general, various, subgroup, or part. Each description must state both what belongs and what is excluded relative to siblings. Before answering, internally check that no two categories are synonyms. Return {"clusters":[{"title":str,"description":str}]} only."""

ASSIGN_SYSTEM = """Assign each reasoning-skill summary to exactly one best child of the taxonomy node below.
Parent path: {path}
Children:
{children}
Return {"assignments":[{"id":integer,"cluster":integer}]} for every item. Use -1 only when no child fits."""

AUDIT_SYSTEM = """Audit whether each programming-reasoning concept fits its assigned taxonomy path and terminal category better than the displayed siblings. Score 1-5: 1 unrelated or a sibling clearly fits better, 3 plausible but broad/overlapping, 5 precise and clearly distinguished. Return {"items":[{"id":integer,"score":integer,"reason":str}]} for every item."""
AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"id": {"type": "integer"}, "score": {"type": "integer", "minimum": 1, "maximum": 5},
                       "reason": {"type": "string", "minLength": 3, "maxLength": 200}},
        "required": ["id", "score", "reason"]}}}, "required": ["items"]}


def leaf_text(row: dict) -> str:
    return f"{row['name']} | invariant: {row['invariant']} | failure: {row['failure_mode']}"


def clamp_words(value: str, limit: int) -> str:
    return " ".join(value.split()[:limit])


def fallback_summaries(text: str) -> list[str]:
    name = text.split(" | ", 1)[0]
    return [clamp_words(name, limit) for limit in LIMITS]


async def summarise_batch(pool: LLMPool, batch: list[tuple[str, str]], counters: Counter) -> tuple[dict, set]:
    user = "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(batch)) + "\nReturn JSON now."
    text, _, error = await pool.chat(
        [{"role": "system", "content": SUM_SYSTEM}, {"role": "user", "content": user}],
        min(8192, 240 * len(batch) + 400), schema=SUM_SCHEMA, schema_name="summaries", temperature=0.0,
    )
    parsed, _ = extract_json(text) if not error else (None, error)
    if parsed is not None and validate_schema_instance(parsed, SUM_SCHEMA):
        parsed = None
    output, fallback = {}, set()
    if parsed and isinstance(parsed.get("items"), list):
        for item in parsed["items"]:
            try:
                index, summaries = int(item["id"]), item["summaries"]
                if not 0 <= index < len(batch) or len(summaries) != SUMMARY_LEVELS:
                    continue
                fixed = [clamp_words(str(value).strip(), LIMITS[level]) for level, value in enumerate(summaries)]
                if all(fixed):
                    output[batch[index][0]] = fixed
            except Exception:
                continue
    missing = [item for item in batch if item[0] not in output]
    if missing:
        counters["summary_retry_items"] += len(missing)
        if len(batch) > 1:
            midpoint = max(1, len(missing) // 2)
            parts = [missing[:midpoint], missing[midpoint:]]
            for result, bad in await asyncio.gather(*(summarise_batch(pool, part, counters) for part in parts if part)):
                output.update(result); fallback |= bad
        else:
            cid, source = missing[0]
            output[cid] = fallback_summaries(source)
            fallback.add(cid)
            counters["summary_fallback"] += 1
    return output, fallback


async def build_summaries(pool: LLMPool, rows: list[dict], counters: Counter, fingerprint: str) -> tuple[dict, set]:
    path = OUTPUT_DIR / "summaries.jsonl"
    cache = {
        row["concept_id"]: row["summaries"] for row in iter_jsonl(path)
        if row.get("quality") == "llm" and row.get("fingerprint") == fingerprint
        and row.get("model_id") == pool.model_signature
        and isinstance(row.get("summaries"), list) and len(row["summaries"]) == SUMMARY_LEVELS
    }
    todo = [(row["concept_id"], leaf_text(row)) for row in rows if row["concept_id"] not in cache]
    print(f"[Summaries] cached={len(cache)} todo={len(todo)}")
    fallback_ids: set[str] = set()
    if todo:
        writer = JsonlWriter(path)
        try:
            batches = [todo[i:i + SUMMARY_BATCH_SIZE] for i in range(0, len(todo), SUMMARY_BATCH_SIZE)]

            async def one(batch):
                result, bad = await summarise_batch(pool, batch, counters)
                for cid, summaries in result.items():
                    if cid not in bad:  # Never make an emergency fallback permanent.
                        writer.write({"concept_id": cid, "summaries": summaries, "quality": "llm",
                                      "fingerprint": fingerprint, "model_id": pool.model_signature})
                return result, bad

            for result, bad in await tqdm.gather(*(one(batch) for batch in batches), desc="Summarising"):
                cache.update(result); fallback_ids |= bad
        finally:
            writer.close()
    return cache, fallback_ids


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.4) + 6


def select_level(leaves: list[str], summaries: dict) -> int:
    fitting = [level for level in range(SUMMARY_LEVELS)
               if sum(estimate_tokens(x) for x in {summaries[cid][level] for cid in leaves}) <= SUMMARY_TOKEN_BUDGET]
    level = max(fitting) if fitting else 0
    while level < SUMMARY_LEVELS - 1 and len({summaries[cid][level] for cid in leaves}) < MIN_UNIQUE_FOR_LEVEL:
        level += 1
    return level


def node_path(state: dict, node: dict) -> list[str]:
    path = []
    current = node
    while current:
        path.append(current["title"])
        parent = current.get("parent")
        current = state["nodes"].get(parent) if parent else None
    return list(reversed(path))


def per_node_rng(node: dict) -> random.Random:
    return random.Random(int(stable_hash(f"{SEED}:{node['id']}:{','.join(sorted(node['leaf_ids']))}", 16), 16))


def weighted_sample_without_replacement(rng: random.Random, items: list[tuple[str, int]], k: int) -> list[tuple[str, int]]:
    """Efraimidis-Spirakis sampling; avoids an unbounded duplicate-draw loop."""
    if len(items) <= k:
        return list(items)
    ranked = []
    for index, (_, weight) in enumerate(items):
        key = rng.random() ** (1.0 / max(1, weight))
        ranked.append((key, index))
    selected = sorted(index for _, index in sorted(ranked, reverse=True)[:k])
    return [items[index] for index in selected]


async def propose(pool: LLMPool, state: dict, node: dict, weighted_items: list[tuple[str, int]], note: str) -> list[dict]:
    lo = min(MIN_BRANCHING, max(2, len(weighted_items) // 3))
    hi = min(MAX_BRANCHING, max(2, len(weighted_items)))
    schema = {"type": "object", "additionalProperties": False, "properties": {"clusters": {
        "type": "array", "minItems": min(lo, hi), "maxItems": hi, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"title": {"type": "string", "minLength": 3, "maxLength": 80},
                           "description": {"type": "string", "minLength": 15, "maxLength": 350}},
            "required": ["title", "description"]}}}, "required": ["clusters"]}
    system = PROPOSE_SYSTEM.format(path=" > ".join(node_path(state, node)), parent=node["title"],
                                   description=node.get("description", ""), lo=lo, hi=hi)
    user = "\n".join(f"- {summary} (represents {count} concepts)" for summary, count in weighted_items)
    if note:
        user += "\nRevision: " + note
    text, _, error = await pool.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     3000, schema=schema, schema_name="clusters", temperature=CLUSTER_TEMPERATURE)
    parsed, _ = extract_json(text) if not error else (None, error)
    if parsed is not None and validate_schema_instance(parsed, schema):
        parsed = None
    if not parsed or not isinstance(parsed.get("clusters"), list):
        return []
    output, seen = [], set()
    forbidden = re.compile(r"\b(misc|other|general|various|subgroup|part)\b", re.I)
    for cluster in parsed["clusters"]:
        title, description = str(cluster.get("title", "")).strip(), str(cluster.get("description", "")).strip()
        if title and description and title.lower() not in seen and not forbidden.search(title):
            seen.add(title.lower()); output.append({"title": title, "description": description})
    return output[:MAX_BRANCHING]


async def assign_batch(pool: LLMPool, state: dict, node: dict, clusters: list[dict], unique: list[str], indices: list[int]) -> dict[int, int]:
    children = "\n".join(f"{i}: {c['title']} — {c['description']}" for i, c in enumerate(clusters))
    schema = {"type": "object", "additionalProperties": False, "properties": {"assignments": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "integer"}, "cluster": {"type": "integer", "minimum": -1,
                                                                    "maximum": len(clusters) - 1}},
            "required": ["id", "cluster"]}}}, "required": ["assignments"]}
    system = ASSIGN_SYSTEM.format(path=" > ".join(node_path(state, node)), children=children)
    user = "\n".join(f"[{i}] {unique[i]}" for i in indices)
    text, _, error = await pool.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     min(8192, 16 * len(indices) + 400), schema=schema, schema_name="assignment", temperature=0.0)
    parsed, _ = extract_json(text) if not error else (None, error)
    if parsed is not None and validate_schema_instance(parsed, schema):
        parsed = None
    result = {}
    if parsed and isinstance(parsed.get("assignments"), list):
        allowed = set(indices)
        for item in parsed["assignments"]:
            try:
                index, cluster = int(item["id"]), int(item["cluster"])
                if index in allowed and -1 <= cluster < len(clusters):
                    result[index] = cluster
            except Exception:
                pass
    return result


async def assign_all(pool: LLMPool, state: dict, node: dict, clusters: list[dict], unique: list[str]) -> tuple[dict, list[int]]:
    indices = list(range(len(unique)))
    batches = [indices[i:i + ASSIGN_BATCH] for i in range(0, len(indices), ASSIGN_BATCH)]
    merged = {}
    for result in await asyncio.gather(*(assign_batch(pool, state, node, clusters, unique, batch) for batch in batches)):
        merged.update({i: c for i, c in result.items() if c >= 0})
    missing = [i for i in indices if i not in merged]
    if missing:
        retries = [missing[i:i + 20] for i in range(0, len(missing), 20)]
        for result in await asyncio.gather(*(assign_batch(pool, state, node, clusters, unique, batch) for batch in retries)):
            merged.update({i: c for i, c in result.items() if c >= 0})
    return merged, [i for i in indices if i not in merged]


async def split_node(pool: LLMPool, state: dict, node: dict, summaries: dict, rows: dict, counters: Counter) -> None:
    leaves = node["leaf_ids"]
    level = select_level(leaves, summaries)
    summary_to_leaves: dict[str, list[str]] = defaultdict(list)
    for cid in leaves:
        summary_to_leaves[summaries[cid][level]].append(cid)
    unique = sorted(summary_to_leaves)
    # De-duplication is useful only while it preserves enough distinctions to split the
    # node. If coarse summaries collapse many leaves, refine to one detailed descriptor
    # per concept rather than arbitrarily chunking an indivisible summary.
    required_groups = max(2, math.ceil(len(leaves) / MAX_TERMINAL_SIZE))
    largest_group = max((len(group) for group in summary_to_leaves.values()), default=0)
    if len(unique) < required_groups or largest_group > MAX_CLUSTER_SHARE * len(leaves):
        summary_to_leaves = {
            (f"[leaf-{stable_hash(cid, 8)}] {summaries[cid][-1]} | "
             f"distinguishing skill: {clamp_words(rows[cid]['name'], 8)} | "
             f"invariant: {clamp_words(rows[cid]['invariant'], 20)}"): [cid]
            for cid in leaves
        }
        if sum(map(len, summary_to_leaves.values())) != len(leaves):
            raise RuntimeError("Leaf descriptor collision during taxonomy refinement")
        unique = sorted(summary_to_leaves)
        counters["leaf_level_refinement"] += 1
    counters[f"summary_level_{level}"] += 1
    node_rng = per_node_rng(node)
    result = None
    note = ""
    for _ in range(BALANCE_RETRIES + 1):
        proposal_items = [(text, len(summary_to_leaves[text])) for text in unique]
        if len(proposal_items) > PROPOSE_MAX_ITEMS:
            proposal_items = weighted_sample_without_replacement(node_rng, proposal_items, PROPOSE_MAX_ITEMS)
        clusters = await propose(pool, state, node, proposal_items, note)
        if len(clusters) < 2:
            counters["proposal_failed"] += 1
            continue
        assignments, missing = await assign_all(pool, state, node, clusters, unique)
        if missing:
            counters["unassigned_summaries"] += len(missing)
            continue
        buckets: dict[int, list[str]] = defaultdict(list)
        for index, cluster in assignments.items():
            buckets[cluster].extend(summary_to_leaves[unique[index]])
        sizes = sorted((len(values), index) for index, values in buckets.items())
        largest = sizes[-1][0]
        allowed_largest = max(MAX_TERMINAL_SIZE, math.ceil(MAX_CLUSTER_SHARE * len(leaves)))
        if len(buckets) >= 2 and largest <= allowed_largest:
            result = (clusters, buckets)
            break
        counters["unbalanced_retry"] += 1
        title = clusters[sizes[-1][1]]["title"] if sizes else "largest category"
        note = f"The prior split put {largest / len(leaves):.0%} in {title!r}. Split the distinctions inside that category."

    children = []
    if result:
        clusters, buckets = result
        for cluster_index in sorted(buckets, key=lambda i: (-len(buckets[i]), clusters[i]["title"])):
            cluster = clusters[cluster_index]
            children.append((cluster["title"], cluster["description"], buckets[cluster_index]))
    else:
        # Do not persist structurally valid but semantically meaningless numbered chunks.
        # The checkpoint from the preceding pass remains resumable, so a later run can retry.
        counters["partition_exhausted"] += 1
        raise RuntimeError(f"Could not obtain a balanced semantic partition for node {node['id']} ({len(leaves)} leaves)")

    node["children"] = []
    for title, description, child_leaves in children:
        child_id = "n_" + stable_hash({"parent": node["id"], "title": title.lower(),
                                        "leaves": sorted(child_leaves)}, 16)
        if child_id in state["nodes"]:
            raise RuntimeError(f"Deterministic taxonomy node-id collision: {child_id}")
        state["nodes"][child_id] = {
            "id": child_id, "title": title, "description": description, "parent": node["id"],
            "depth": node["depth"] + 1, "children": [], "leaf_ids": child_leaves, "state": "pending",
        }
        node["children"].append(child_id)
    node["leaf_ids"] = []
    node["state"] = "done"


def save_state(state: dict) -> None:
    atomic_write_json(OUTPUT_DIR / "tree_state.json", state, indent=None)


async def build_tree(pool: LLMPool, leaves: list[str], summaries: dict, rows: dict, counters: Counter, fingerprint: str) -> dict:
    state_path = OUTPUT_DIR / "tree_state.json"
    if state_path.exists():
        candidate = json.loads(state_path.read_text(encoding="utf-8"))
        if candidate.get("fingerprint") == fingerprint:
            state = candidate
            print(f"[Tree] Resuming {len(state['nodes'])} nodes")
        else:
            print("[Tree] Existing checkpoint fingerprint is stale; starting a new state")
            state = {}
    else:
        state = {}
    if not state:
        state = {"fingerprint": fingerprint, "next_id": 1, "nodes": {"n0": {
            "id": "n0", "title": ROOT_TITLE, "description": ROOT_DESCRIPTION, "parent": None,
            "depth": 0, "children": [], "leaf_ids": list(leaves), "state": "pending",
        }}}
    pass_number = 0
    while True:
        changed = False
        for node in state["nodes"].values():
            if node["state"] == "pending" and len(node["leaf_ids"]) <= MAX_TERMINAL_SIZE:
                node["state"] = "done"; changed = True
        frontier = [node for node in state["nodes"].values() if node["state"] == "pending"]
        if not frontier:
            if changed:
                save_state(state)
            break
        pass_number += 1
        print(f"[Tree] pass={pass_number} nodes={len(frontier)} largest={max(len(n['leaf_ids']) for n in frontier)}")
        await tqdm.gather(*(split_node(pool, state, node, summaries, rows, counters) for node in frontier),
                          desc=f"Taxonomy pass {pass_number}")
        save_state(state)
    return state


def path_for(state: dict, node_id: str) -> tuple[list[str], list[str]]:
    ids, titles = [], []
    while node_id:
        node = state["nodes"][node_id]
        ids.append(node_id); titles.append(node["title"])
        node_id = node["parent"]
    return list(reversed(ids)), list(reversed(titles))


async def audit_taxonomy(pool: LLMPool, state: dict, rows: dict, counters: Counter) -> list[dict]:
    terminal_pairs = sorted((cid, node["id"]) for node in state["nodes"].values()
                            if not node["children"] for cid in node["leaf_ids"])
    rng = random.Random(SEED + 99)
    sampled = rng.sample(terminal_pairs, min(AUDIT_SAMPLE_SIZE, len(terminal_pairs)))
    batches = [sampled[i:i + 15] for i in range(0, len(sampled), 15)]
    async def one(batch):
        user_lines = []
        for index, (cid, node_id) in enumerate(batch):
            _, path = path_for(state, node_id)
            node = state["nodes"][node_id]
            parent = state["nodes"].get(node.get("parent"))
            siblings = [] if parent is None else [
                f"{state['nodes'][sibling]['title']}: {state['nodes'][sibling].get('description', '')[:160]}"
                for sibling in parent["children"]
            ]
            user_lines.append(
                f"[{index}] concept: {leaf_text(rows[cid])}\nassigned path: {' > '.join(path)}\n"
                f"terminal siblings: {' || '.join(siblings)}"
            )
        text, _, error = await pool.chat(
            [{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": "\n\n".join(user_lines)}],
            2500, schema=AUDIT_SCHEMA, schema_name="taxonomy_audit", temperature=0.0,
        )
        parsed, _ = extract_json(text) if not error else (None, error)
        if parsed is not None and validate_schema_instance(parsed, AUDIT_SCHEMA):
            parsed = None
        by_index = {}
        if parsed:
            for item in parsed.get("items", []):
                try:
                    index, score, reason = int(item["id"]), int(item["score"]), str(item["reason"]).strip()
                    if 0 <= index < len(batch) and 1 <= score <= 5 and reason and index not in by_index:
                        cid, node_id = batch[index]
                        by_index[index] = {"concept_id": cid, "node_id": node_id, "score": score, "reason": reason}
                except Exception:
                    pass
        missing = len(batch) - len(by_index)
        if missing:
            counters["taxonomy_audit_missing"] += missing
        return [by_index[index] for index in sorted(by_index)]

    rows_out = []
    for result in await tqdm.gather(*(one(batch) for batch in batches), desc="Auditing taxonomy"):
        rows_out.extend(result)
    return rows_out


def export(state: dict, rows: dict) -> dict:
    nodes = state["nodes"]

    def nested(node_id: str):
        node = nodes[node_id]
        result = {"id": node_id, "title": node["title"], "description": node.get("description", ""), "depth": node["depth"]}
        if node["children"]:
            result["children"] = [nested(child) for child in node["children"]]
            result["size"] = sum(child["size"] for child in result["children"])
        else:
            result["size"] = len(node["leaf_ids"])
            result["examples"] = [rows[cid]["name"] for cid in node["leaf_ids"][:5]]
        return result

    tree = nested("n0")
    atomic_write_json(OUTPUT_DIR / "taxonomy_tree.json", tree, indent=1)
    node_rows, assignment_rows, seen = [], [], Counter()
    for node_id in sorted(nodes):
        node = nodes[node_id]
        ids, titles = path_for(state, node_id)
        terminal = not node["children"]
        node_rows.append({"node_id": node_id, "parent_id": node["parent"], "depth": node["depth"],
                          "title": node["title"], "description": node.get("description", ""),
                          "path": titles[1:], "path_ids": ids[1:], "terminal": terminal,
                          "n_children": len(node["children"]), "n_leaves": len(node["leaf_ids"]) if terminal else None})
        if terminal:
            for cid in node["leaf_ids"]:
                seen[cid] += 1
                assignment_rows.append({"concept_id": cid, "node_id": node_id, "depth": node["depth"],
                                        "path": titles[1:], "path_ids": ids[1:]})
    from llm_pool import atomic_write_jsonl
    atomic_write_jsonl(OUTPUT_DIR / "taxonomy_nodes.jsonl", node_rows)
    atomic_write_jsonl(OUTPUT_DIR / "taxonomy_leaf_assignments.jsonl", assignment_rows)
    problems = []
    if set(seen) != set(rows):
        problems.append("leaf coverage mismatch")
    if any(count != 1 for count in seen.values()):
        problems.append("leaves assigned more than once")
    if any(len(node["children"]) > MAX_BRANCHING for node in nodes.values()):
        problems.append("branching limit exceeded")
    if any(not node["children"] and len(node["leaf_ids"]) > MAX_TERMINAL_SIZE for node in nodes.values()):
        problems.append("terminal size exceeded")
    terminals = [node for node in nodes.values() if not node["children"]]
    return {"n_leaves": len(rows), "n_nodes": len(nodes), "n_terminal": len(terminals),
            "max_depth": max(node["depth"] for node in nodes.values()),
            "terminal_size_histogram": dict(Counter(len(node["leaf_ids"]) for node in terminals)),
            "top_level": [(child["title"], child["size"]) for child in tree.get("children", [])],
            "validation_problems": problems}


async def amain() -> int:
    (OUTPUT_DIR / "_SUCCESS").unlink(missing_ok=True)
    if not (AUTOPSY_DIR / "_SUCCESS").exists():
        print("[Fatal] Stage 1 has not passed stage1_verify.py")
        return 2
    rows_by_id = {}
    for row in iter_jsonl(CONCEPTS_FILE, strict=True):
        if row.get("concept_id") not in rows_by_id:
            rows_by_id[row["concept_id"]] = row
    if LEAF_LIMIT:
        rows_by_id = dict(list(rows_by_id.items())[:LEAF_LIMIT])
    if not rows_by_id:
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_fingerprint = stable_hash([(cid, leaf_text(row)) for cid, row in sorted(rows_by_id.items())])
    code_fingerprint = stable_hash({
        name: file_hash(ROOT / name) for name in ("stage2_taxonomy.py", "llm_pool.py", "pipeline_config.py")
    })
    summary_fingerprint = stable_hash({"input": input_fingerprint, "prompt": SUM_SYSTEM,
                                       "levels": LIMITS, "pipeline": PIPELINE_VERSION,
                                       "code": code_fingerprint})
    tree_fingerprint = ""  # completed after summaries are available
    counters = Counter(); start = time.time()
    connector = aiohttp.TCPConnector(limit=GLOBAL_CONCURRENCY + 50)
    async with aiohttp.ClientSession(connector=connector) as session:
        pool = LLMPool(SERVERS, session, concurrency=GLOBAL_CONCURRENCY, timeout=HTTP_REQUEST_TIMEOUT,
                       max_retries=MAX_HTTP_RETRIES, temperature=CLUSTER_TEMPERATURE, mode=STRUCTURED_OUTPUT_MODE,
                       extra_body=EXTRA_REQUEST_BODY, served_model_id=SERVED_MODEL_ID,
                       allow_mixed_models=ALLOW_MIXED_MODELS, errors=counters)
        await pool.startup(SUM_SCHEMA)
        summaries, fallback_ids = await build_summaries(pool, list(rows_by_id.values()), counters, summary_fingerprint)
        if fallback_ids:
            print(f"[Failed] {len(fallback_ids)} concepts exhausted summary retries. Fallbacks were not cached; rerun to retry them.")
            return 1
        leaves = [cid for cid in rows_by_id if cid in summaries]
        tree_fingerprint = stable_hash({
            "input": input_fingerprint, "summary_config": summary_fingerprint,
            "summary_content": [(cid, summaries[cid]) for cid in sorted(leaves)],
            "branch": MAX_BRANCHING, "terminal": MAX_TERMINAL_SIZE,
            "prompts": [PROPOSE_SYSTEM, ASSIGN_SYSTEM], "model_id": pool.model_signature,
            "code": code_fingerprint,
        })
        state = await build_tree(pool, leaves, summaries, rows_by_id, counters, tree_fingerprint)

    # A separate judge endpoint/model can be configured for semantic path auditing.
    judge_connector = aiohttp.TCPConnector(limit=GLOBAL_CONCURRENCY + 50)
    async with aiohttp.ClientSession(connector=judge_connector) as judge_session:
        judge_pool = LLMPool(JUDGE_SERVERS, judge_session, concurrency=GLOBAL_CONCURRENCY,
                             timeout=HTTP_REQUEST_TIMEOUT, max_retries=MAX_HTTP_RETRIES, temperature=0.0,
                             mode=STRUCTURED_OUTPUT_MODE, extra_body=EXTRA_REQUEST_BODY,
                             served_model_id=JUDGE_MODEL_ID, allow_mixed_models=ALLOW_MIXED_MODELS, errors=counters)
        await judge_pool.startup(AUDIT_SCHEMA)
        if REQUIRE_DISTINCT_JUDGE_MODEL and (set(judge_pool.model_ids) & set(pool.model_ids)):
            raise RuntimeError("Taxonomy audit requires a judge model distinct from the clustering model")
        audit_rows = await audit_taxonomy(judge_pool, state, rows_by_id, counters)
    from llm_pool import atomic_write_jsonl
    atomic_write_jsonl(OUTPUT_DIR / "taxonomy_audit.jsonl", audit_rows)

    report = export(state, {cid: rows_by_id[cid] for cid in leaves})
    scores = [row["score"] for row in audit_rows]
    report["semantic_audit"] = {
        "n": len(scores), "mean": round(sum(scores) / max(1, len(scores)), 3),
        "low_fit_share": round(sum(score < 3 for score in scores) / max(1, len(scores)), 3),
    }
    if counters["taxonomy_audit_missing"]:
        report["validation_problems"].append("taxonomy semantic audit was incomplete")
    if report["semantic_audit"]["mean"] < MIN_TAXONOMY_AUDIT_MEAN or report["semantic_audit"]["low_fit_share"] > MAX_LOW_FIT_SHARE:
        report["validation_problems"].append("taxonomy semantic audit did not meet quality thresholds")
    report.update({"counters": dict(counters), "input_fingerprint": input_fingerprint,
                   "tree_fingerprint": tree_fingerprint, "seconds": round(time.time() - start, 1),
                   "clustering_model": pool.model_signature, "judge_model": judge_pool.model_signature,
                   "independent_judge": not (set(pool.model_ids) & set(judge_pool.model_ids)),
                   "global_concurrency": GLOBAL_CONCURRENCY})
    atomic_write_json(OUTPUT_DIR / "taxonomy_report.json", report)
    if report["validation_problems"]:
        print(f"[Failed] {report['validation_problems']}")
        return 1
    (OUTPUT_DIR / "_SUCCESS").write_text("verified\n", encoding="utf-8")
    print(f"[Verified] {report['n_leaves']} leaves, {report['n_terminal']} terminal cells")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
