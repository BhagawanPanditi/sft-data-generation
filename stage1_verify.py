#!/usr/bin/env python3
"""Stage 1 hard gate: integrity, evidence grounding, leakage and distribution checks."""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp

from llm_pool import (
    LLMPool, atomic_write_json, atomic_write_jsonl, extract_json, load_module, read_jsonl,
    validate_schema_instance,
)
from pipeline_config import (
    ALLOW_MIXED_MODELS, AUTOPSY_DIR as OUTPUT_DIR, DATA_DIR, EXTRA_REQUEST_BODY,
    GLOBAL_CONCURRENCY, HTTP_REQUEST_TIMEOUT, JUDGE_MODEL_ID, JUDGE_SERVERS,
    MAX_HTTP_RETRIES, REQUIRE_DISTINCT_JUDGE_MODEL, ROOT, STRUCTURED_OUTPUT_MODE,
)

X = load_module(ROOT / "stage1_extract.py", "stage1_extract_for_verify")
REPORT_DIR = OUTPUT_DIR / "verification"
SEED = 13
EYEBALL_PER_FAMILY = 3
STRICT_OVERLAP = 0.20
MIN_COVERAGE = 0.85
VAGUE = re.compile(r"\b(mishandl\w*|edge cases?|incorrectly|improperly|fails? to handle|various|some cases|certain inputs)\b", re.I)
MECHANISM = re.compile(r"\b(because|instead|only|never|before|after|without|ignores?|returns?|assumes?|uses?|skips?|stops?|overwrites?|mutates?|initializes?|compares?|resets?)\b", re.I)
FALSIFIABLE = re.compile(r"\b(must|always|never|every|all|only|exactly|at least|at most|equal|preserv\w*|remain\w*)\b", re.I)
JUDGE_SAMPLE_SIZE = 300
JUDGE_MAX_SOURCE_CHARS = 25_000
JUDGE_SYSTEM = """Independently audit a programming-problem autopsy against its source material. Be skeptical.
Score 1-5 for: faithful (all claims supported), transferable (not a task restatement), failure_concrete
(specific wrong mechanism + exposing input class + observable consequence), invariant_falsifiable, and
key_concept_coverage. Also decide whether each evidence quote supports its concept and whether every basis label
matches the section type. Return one JSON object only."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "faithful": {"type": "integer", "minimum": 1, "maximum": 5},
    "transferable": {"type": "integer", "minimum": 1, "maximum": 5},
    "failure_concrete": {"type": "integer", "minimum": 1, "maximum": 5},
    "invariant_falsifiable": {"type": "integer", "minimum": 1, "maximum": 5},
    "key_concept_coverage": {"type": "integer", "minimum": 1, "maximum": 5},
    "evidence_supports": {"type": "boolean"}, "basis_correct": {"type": "boolean"},
    "comment": {"type": "string", "minLength": 3, "maxLength": 300},
}, "required": ["faithful", "transferable", "failure_concrete", "invariant_falsifiable",
                "key_concept_coverage", "evidence_supports", "basis_correct", "comment"]}


def canonical_views(audit: list[dict]) -> tuple[list[dict], list[dict]]:
    clean_rows, flat_rows = [], []
    for record in audit:
        clean_concepts = []
        for rank, concept in enumerate(record.get("concepts", []), 1):
            clean = {k: v for k, v in concept.items() if k not in {"evidence", "evidence_section"}}
            clean_concepts.append(clean)
            flat_rows.append({k: v for k, v in record.items() if k != "concepts"} | clean | {"concept_rank": rank})
        clean_rows.append({k: v for k, v in record.items() if k != "concepts"} | {"concepts": clean_concepts})
    return clean_rows, flat_rows


def keyed(rows: list[dict], key: str) -> dict:
    return {str(row[key]): row for row in rows if key in row}


def integrity(audit, clean, flat, rejected, failed) -> tuple[dict, list[str]]:
    expected_clean, expected_flat = canonical_views(audit)
    audit_ids = [str(x.get("sample_id")) for x in audit]
    clean_ids = [str(x.get("sample_id")) for x in clean]
    concept_ids = [str(c.get("concept_id")) for row in audit for c in row.get("concepts", [])]
    flat_ids = [str(x.get("concept_id")) for x in flat]
    clean_by_id, expected_clean_by_id = keyed(clean, "sample_id"), keyed(expected_clean, "sample_id")
    flat_by_id, expected_flat_by_id = keyed(flat, "concept_id"), keyed(expected_flat, "concept_id")
    problems = []
    if len(audit_ids) != len(set(audit_ids)):
        problems.append("duplicate sample ids in audit")
    if len(concept_ids) != len(set(concept_ids)):
        problems.append("duplicate concept ids in audit")
    if len(clean_ids) != len(set(clean_ids)):
        problems.append("duplicate sample ids in clean view")
    if len(flat_ids) != len(set(flat_ids)):
        problems.append("duplicate concept ids in flat view")
    if any("sample_id" not in row for row in clean) or any("concept_id" not in row for row in flat):
        problems.append("clean or flat view contains rows without primary keys")
    if clean_by_id != expected_clean_by_id:
        problems.append("clean view is not an exact projection of audit")
    if flat_by_id != expected_flat_by_id:
        problems.append("flat view is not an exact projection of audit")
    both = set(audit_ids) & {str(x.get("sample_id")) for x in rejected}
    if both:
        problems.append(f"{len(both)} ids occur in both audit and rejected")
    result = {
        "audit_records": len(audit), "clean_records": len(clean), "flat_records": len(flat),
        "rejected_records": len(rejected), "failed_attempts": len(failed),
        "audit_sample_ids": len(set(audit_ids)), "audit_concept_ids": len(set(concept_ids)),
        "exact_clean_projection": clean_by_id == expected_clean_by_id,
        "exact_flat_projection": flat_by_id == expected_flat_by_id,
        "problems": problems,
    }
    return result, problems


def autopsy_from_record(record: dict):
    obj = {key: record[key] for key in
           ("topics", "primary_failure_mode", "difficulty_syntax", "difficulty_reasoning") if key in record}
    obj["concepts"] = [{k: v for k, v in c.items() if k not in {"concept_id", "failure_mode"}}
                       for c in record.get("concepts", [])]
    return X.Autopsy.model_validate(obj)


def check_records(audit: list[dict], samples: dict[str, X.Sample]):
    flags: dict[str, list[tuple[str, str]]] = defaultdict(list)
    counts = Counter()
    for record in audit:
        sid = str(record.get("sample_id"))
        sample = samples.get(sid)
        if sample is None:
            flags[sid].append(("hard", "source_missing")); counts["source_missing"] += 1
            continue
        counts["problems"] += 1
        try:
            autopsy = autopsy_from_record(record)
        except Exception as exc:
            flags[sid].append(("hard", f"schema:{type(exc).__name__}")); counts["schema"] += 1
            continue
        issue = X.validate_concepts(autopsy, sample)
        if issue:
            flags[sid].append(("hard", f"grounding:{issue}")); counts["grounding"] += 1
        names = set()
        for concept in autopsy.concepts:
            counts["concepts"] += 1
            name = re.sub(r"\W+", " ", concept.name.lower()).strip()
            if name in names:
                flags[sid].append(("hard", "duplicate_concepts")); counts["duplicate_concepts"] += 1
            names.add(name)
            words = concept.name.split()
            if not 3 <= len(words) <= 8:
                flags[sid].append(("soft", f"name_length:{len(words)}")); counts["name_length"] += 1
            if not FALSIFIABLE.search(concept.invariant):
                flags[sid].append(("soft", "invariant_not_falsifiable")); counts["invariant_not_falsifiable"] += 1
            if re.search(r"[.!?]\s+[A-Z]", concept.invariant):
                flags[sid].append(("soft", "multi_sentence_invariant")); counts["multi_sentence_invariant"] += 1
            failure = concept.failure_mode
            if VAGUE.search(failure):
                flags[sid].append(("soft", "vague_failure")); counts["vague_failure"] += 1
            if not MECHANISM.search(failure):
                flags[sid].append(("soft", "failure_without_mechanism")); counts["failure_without_mechanism"] += 1
        leak = X.leak_reason(autopsy, sample)
        if leak:
            flags[sid].append(("hard", f"leak:{leak}")); counts["leak"] += 1
        text = " ".join([record.get("primary_failure_mode", ""), *record.get("topics", [])] + [
            " ".join(str(c.get(k, "")) for k in ("name", "invariant", "wrong_mechanism", "failing_input_class", "observable_failure"))
            for c in record.get("concepts", [])
        ])
        mine, source = X.ngrams(text), X.ngrams(sample.payload)
        ratio = len(mine & source) / max(1, len(mine))
        if STRICT_OVERLAP < ratio <= 0.30:
            flags[sid].append(("soft", f"strict_overlap:{ratio:.2f}")); counts["strict_overlap"] += 1
    return flags, counts


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum((v / total) * math.log2(v / total) for v in counter.values()) if total else 0.0


def distributions(audit: list[dict], rejected: list[dict], failed: list[dict]) -> dict:
    topics = Counter(t for row in audit for t in row.get("topics", []))
    names = Counter(c["name"].strip().lower() for row in audit for c in row.get("concepts", []))
    basis = Counter(c["basis"] for row in audit for c in row.get("concepts", []))
    family_ok = Counter(row.get("family") for row in audit)
    family_rej = Counter(row.get("family") for row in rejected)
    family_fail = Counter(row.get("family") for row in failed)
    return {
        "yield_by_family": {family: {
            "ok": family_ok[family], "rejected": family_rej[family], "failed_attempts": family_fail[family],
            "yield": round(family_ok[family] / max(1, family_ok[family] + family_rej[family]), 3),
        } for family in sorted(set(family_ok) | set(family_rej) | set(family_fail))},
        "difficulty_syntax": dict(sorted(Counter(x["difficulty_syntax"] for x in audit).items())),
        "difficulty_reasoning": dict(sorted(Counter(x["difficulty_reasoning"] for x in audit).items())),
        "basis": dict(basis), "concept_name_unique_ratio": round(len(names) / max(1, sum(names.values())), 3),
        "top_repeated_names": names.most_common(20),
        "topics": {"distinct": len(topics), "top": topics.most_common(30),
                   "singleton_share": round(sum(v == 1 for v in topics.values()) / max(1, len(topics)), 3),
                   "normalized_entropy": round(entropy(topics) / math.log2(max(2, len(topics))), 3)},
    }


def evidence_context(section: X.Section, evidence: str, radius: int = 350) -> str:
    first_piece = next((piece.strip() for piece in re.split(r"\.\.\.|…", evidence) if piece.strip()), evidence)
    pos = section.text.find(first_piece)
    if pos < 0:
        return section.text[:radius * 2]
    return section.text[max(0, pos - radius):pos + len(first_piece) + radius]


def write_eyeball(audit: list[dict], samples: dict[str, X.Sample], path: Path) -> None:
    rng = random.Random(SEED)
    by_family = defaultdict(list)
    for row in audit:
        by_family[row["family"]].append(row)
    lines = []
    for family, rows in sorted(by_family.items()):
        for row in rng.sample(rows, min(EYEBALL_PER_FAMILY, len(rows))):
            sample = samples.get(row["sample_id"])
            lines.extend(["=" * 100, f"{family} | {row['sample_id']} | topics={row.get('topics')}"])
            for concept in row["concepts"]:
                lines.append(f"[{concept['basis']}] {concept['name']}")
                lines.append(f"evidence ({concept['evidence_section']}): {concept['evidence']}")
                if sample:
                    section = next((s for s in sample.sections if s.id == concept["evidence_section"]), None)
                    if section:
                        lines.append("context: " + evidence_context(section, concept["evidence"]))
                lines.append("invariant: " + concept["invariant"])
                lines.append("failure: " + concept["failure_mode"])
    path.write_text("\n".join(lines), encoding="utf-8")


def judge_sample(audit: list[dict]) -> list[dict]:
    """Deterministically stratify the audit across benchmark families."""
    rng = random.Random(SEED + 101)
    by_family = defaultdict(list)
    for row in audit:
        by_family[row["family"]].append(row)
    selected = []
    families = sorted(by_family)
    per_family = max(1, math.ceil(JUDGE_SAMPLE_SIZE / max(1, len(families))))
    for family in families:
        rows = by_family[family]
        selected.extend(rng.sample(rows, min(per_family, len(rows))))
    if len(selected) > JUDGE_SAMPLE_SIZE:
        selected = rng.sample(selected, JUDGE_SAMPLE_SIZE)
    return sorted(selected, key=lambda row: row["sample_id"])


async def run_llm_judge(rows: list[dict], samples: dict[str, X.Sample]) -> tuple[list[dict], str]:
    connector = aiohttp.TCPConnector(limit=GLOBAL_CONCURRENCY + 50)
    async with aiohttp.ClientSession(connector=connector) as session:
        pool = LLMPool(
            JUDGE_SERVERS, session, concurrency=GLOBAL_CONCURRENCY, timeout=HTTP_REQUEST_TIMEOUT,
            max_retries=MAX_HTTP_RETRIES, temperature=0.0, mode=STRUCTURED_OUTPUT_MODE,
            extra_body=EXTRA_REQUEST_BODY, served_model_id=JUDGE_MODEL_ID,
            allow_mixed_models=ALLOW_MIXED_MODELS,
        )
        await pool.startup(JUDGE_SCHEMA)
        extraction_models = {model for row in rows for model in str(row.get("model_id", "")).split(",") if model}
        if REQUIRE_DISTINCT_JUDGE_MODEL and (set(pool.model_ids) & extraction_models):
            raise RuntimeError("Stage-1 verifier requires a judge model distinct from the extraction model")

        async def one(row: dict) -> dict:
            sample = samples[row["sample_id"]]
            shown = {k: v for k, v in row.items() if k not in {"model_id"}}
            if len(sample.payload) <= JUDGE_MAX_SOURCE_CHARS:
                source_for_judge = sample.payload
            else:
                excerpts = [
                    f"### [{section.id} | {section.kind} | {section.label}]\n{section.text[:1800]}"
                    for section in sample.sections if section.kind in {"statement", "starter_code", "trace_program"}
                ]
                by_id = {section.id: section for section in sample.sections}
                for concept in row.get("concepts", []):
                    section = by_id.get(concept.get("evidence_section"))
                    if section:
                        excerpts.append(
                            f"### [{section.id} | {section.kind} | {section.label}]\n"
                            + evidence_context(section, concept.get("evidence", ""), radius=1200)
                        )
                source_for_judge = "\n\n".join(excerpts)[:JUDGE_MAX_SOURCE_CHARS]
            user = (f"SOURCE MATERIAL\n{source_for_judge}\n\n"
                    f"AUTOPSY\n{json.dumps(shown, ensure_ascii=False)}\n\nReturn JSON now.")
            text, _, error = await pool.chat(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                1200, schema=JUDGE_SCHEMA, schema_name="autopsy_audit", temperature=0.0,
            )
            parsed, parse_error = extract_json(text) if not error else (None, error)
            schema_error = validate_schema_instance(parsed, JUDGE_SCHEMA) if parsed is not None else None
            if parsed is None or schema_error:
                return {"sample_id": row["sample_id"], "family": row["family"],
                        "error": schema_error or parse_error or "invalid judge response"}
            return {"sample_id": row["sample_id"], "family": row["family"], **parsed}

        judged = await asyncio.gather(*(one(row) for row in rows))
        return judged, pool.model_signature


def summarize_llm_judge(rows: list[dict]) -> dict:
    valid = [row for row in rows if "error" not in row]
    result = {"requested": len(rows), "valid": len(valid), "errors": len(rows) - len(valid)}
    score_fields = ("faithful", "transferable", "failure_concrete", "invariant_falsifiable", "key_concept_coverage")
    for field in score_fields:
        values = [int(row[field]) for row in valid]
        result[f"mean_{field}"] = round(sum(values) / max(1, len(values)), 3)
        result[f"low_share_{field}"] = round(sum(value < 3 for value in values) / max(1, len(values)), 3)
    result["evidence_failure_share"] = round(sum(not row["evidence_supports"] for row in valid) / max(1, len(valid)), 3)
    result["basis_failure_share"] = round(sum(not row["basis_correct"] for row in valid) / max(1, len(valid)), 3)
    return result


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "_SUCCESS").unlink(missing_ok=True)
    try:
        audit = read_jsonl(OUTPUT_DIR / "autopsies.jsonl", strict=True)
        clean = read_jsonl(OUTPUT_DIR / "autopsies_clean.jsonl", strict=True)
        flat = read_jsonl(OUTPUT_DIR / "concepts_flat.jsonl", strict=True)
        rejected = read_jsonl(OUTPUT_DIR / "rejected.jsonl", strict=True)
        failed = read_jsonl(OUTPUT_DIR / "failed.jsonl", strict=True)
    except Exception as exc:
        print(f"[Fatal] Could not read outputs strictly: {exc}")
        return 2
    if not audit:
        print("[Fatal] No autopsies")
        return 2

    stats = X.Stats()
    files = []
    for path in sorted(p for p in DATA_DIR.iterdir() if p.suffix in {".json", ".jsonl"}):
        try:
            files.append((path, X.route(path.stem)))
        except ValueError:
            pass
    samples = {sample.sample_id: sample for sample in X.load_samples(files, stats)}
    integ, integrity_problems = integrity(audit, clean, flat, rejected, failed)
    flags, counts = check_records(audit, samples)
    eligible = len(samples)
    coverage = len({row["sample_id"] for row in audit}) / max(1, eligible)
    n_concepts = max(1, counts["concepts"])
    n_problems = max(1, counts["problems"])
    distributions_report = distributions(audit, rejected, failed)
    basis_counts = Counter(c["basis"] for row in audit for c in row.get("concepts", []))
    hypothesis_share = basis_counts["hypothesis"] / max(1, sum(basis_counts.values()))
    syntax_hist = Counter(row["difficulty_syntax"] for row in audit)
    reasoning_hist = Counter(row["difficulty_reasoning"] for row in audit)
    hard_rows = {sid for sid, values in flags.items() if any(severity == "hard" for severity, _ in values)}
    scorecard = [
        ("coverage", coverage, MIN_COVERAGE, coverage >= MIN_COVERAGE),
        ("hard flagged share", len(hard_rows) / n_problems, 0.0, not hard_rows),
        ("vague failure share", counts["vague_failure"] / n_concepts, 0.10, counts["vague_failure"] / n_concepts <= 0.10),
        ("failure missing mechanism share", counts["failure_without_mechanism"] / n_concepts, 0.20, counts["failure_without_mechanism"] / n_concepts <= 0.20),
        ("strict phrase overlap share", counts["strict_overlap"] / n_problems, 0.05, counts["strict_overlap"] / n_problems <= 0.05),
        ("bad concept-name length share", counts["name_length"] / n_concepts, 0.05, counts["name_length"] / n_concepts <= 0.05),
        ("non-falsifiable invariant share", counts["invariant_not_falsifiable"] / n_concepts, 0.15,
         counts["invariant_not_falsifiable"] / n_concepts <= 0.15),
        ("multi-sentence invariant share", counts["multi_sentence_invariant"] / n_concepts, 0.05,
         counts["multi_sentence_invariant"] / n_concepts <= 0.05),
        ("hypothesis basis share", hypothesis_share, 0.35, hypothesis_share <= 0.35),
        ("concept-name unique ratio", distributions_report["concept_name_unique_ratio"], 0.40,
         distributions_report["concept_name_unique_ratio"] >= 0.40),
        ("syntax difficulty max-bin share", max(syntax_hist.values()) / max(1, len(audit)), 0.60,
         max(syntax_hist.values()) / max(1, len(audit)) <= 0.60),
        ("reasoning difficulty max-bin share", max(reasoning_hist.values()) / max(1, len(audit)), 0.60,
         max(reasoning_hist.values()) / max(1, len(audit)) <= 0.60),
    ]
    judge_rows = []
    judge_model = ""
    try:
        sampled = judge_sample([row for row in audit if row.get("sample_id") in samples])
        judge_rows, judge_model = asyncio.run(run_llm_judge(sampled, samples))
        judge_summary = summarize_llm_judge(judge_rows)
    except Exception as exc:
        judge_summary = {"requested": JUDGE_SAMPLE_SIZE, "valid": 0, "errors": JUDGE_SAMPLE_SIZE,
                         "fatal_error": f"{type(exc).__name__}: {exc}"}
    atomic_write_jsonl(REPORT_DIR / "judge.jsonl", judge_rows)
    scorecard.extend([
        ("LLM judge response coverage", judge_summary.get("valid", 0) / max(1, judge_summary.get("requested", 1)),
         1.0, judge_summary.get("errors", 1) == 0),
        ("LLM faithful mean", judge_summary.get("mean_faithful", 0.0), 3.8,
         judge_summary.get("mean_faithful", 0.0) >= 3.8),
        ("LLM transferable mean", judge_summary.get("mean_transferable", 0.0), 3.7,
         judge_summary.get("mean_transferable", 0.0) >= 3.7),
        ("LLM concrete-failure mean", judge_summary.get("mean_failure_concrete", 0.0), 3.7,
         judge_summary.get("mean_failure_concrete", 0.0) >= 3.7),
        ("LLM falsifiable-invariant mean", judge_summary.get("mean_invariant_falsifiable", 0.0), 3.7,
         judge_summary.get("mean_invariant_falsifiable", 0.0) >= 3.7),
        ("LLM evidence failure share", judge_summary.get("evidence_failure_share", 1.0), 0.05,
         judge_summary.get("evidence_failure_share", 1.0) <= 0.05),
        ("LLM basis failure share", judge_summary.get("basis_failure_share", 1.0), 0.05,
         judge_summary.get("basis_failure_share", 1.0) <= 0.05),
    ])

    flagged_rows = [{
        "sample_id": sid, "family": samples[sid].family if sid in samples else "?",
        "severity": "hard" if sid in hard_rows else "soft", "reasons": sorted({reason for _, reason in values}),
    } for sid, values in sorted(flags.items())]
    atomic_write_jsonl(REPORT_DIR / "flagged.jsonl", flagged_rows)
    write_eyeball(audit, samples, REPORT_DIR / "eyeball.txt")
    report = {
        "integrity": integ, "coverage": {"eligible": eligible, "autopsied": len(audit), "ratio": round(coverage, 4)},
        "checks": dict(counts), "flagged": {"hard": len(hard_rows), "total": len(flags)},
        "distributions": distributions_report,
        "llm_judge": {**judge_summary, "model_id": judge_model,
                      "independent": not (set(judge_model.split(",")) & {
                          model for row in audit for model in str(row.get("model_id", "")).split(",") if model
                      })},
        "scorecard": [{"check": name, "value": round(value, 4), "limit": limit, "status": "ok" if ok else "FAIL"}
                      for name, value, limit, ok in scorecard],
    }
    atomic_write_json(REPORT_DIR / "report.json", report)
    passed = not integrity_problems and all(row[3] for row in scorecard)
    print("\nSCORECARD")
    for name, value, limit, ok in scorecard:
        print(f"  [{'ok' if ok else 'FAIL':<4}] {name:<34} {value:.4f} (limit {limit})")
    if integrity_problems:
        print(f"  [FAIL] integrity: {integrity_problems}")
    if passed:
        (OUTPUT_DIR / "_SUCCESS").write_text("verified\n", encoding="utf-8")
        print(f"[Verified] Marker written to {OUTPUT_DIR / '_SUCCESS'}")
        return 0
    print(f"[Failed] Inspect {REPORT_DIR / 'flagged.jsonl'} and eyeball.txt; stage 2 remains blocked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
