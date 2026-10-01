#!/usr/bin/env python3
"""Download and normalize the benchmark sources used by the pipeline.

Dataset revisions are resolved to immutable Hugging Face commit SHAs and recorded in
``data/manifests/download_manifest.json``. LBPP's encoded code fields are decoded with a
restricted unpickler; undecodable records are rejected rather than converted into fake solutions.
"""
from __future__ import annotations

import os

from datasets import load_dataset
from huggingface_hub import HfApi

from dataset_codecs import decode_lbpp_value
from llm_pool import atomic_write_json, atomic_write_jsonl, file_hash, iter_jsonl
from pipeline_config import DATA_DIR, PIPELINE_VERSION, ROOT

MANIFEST_DIR = ROOT / "data" / "manifests"
REJECT_DIR = ROOT / "data" / "rejected"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_revision(repo_id: str) -> str:
    env_name = "HF_REVISION_" + repo_id.upper().replace("/", "_").replace("-", "_")
    requested = os.getenv(env_name)
    if requested:
        return requested
    return HfApi().dataset_info(repo_id).sha


def save_dataset(repo_id: str, filename: str, *, config: str | None = None,
                 split: str = "test", required: set[str] = frozenset(),
                 required_any: tuple[str, ...] = ()) -> dict:
    revision = resolve_revision(repo_id)
    print(f"[Download] {repo_id} config={config!r} split={split!r} revision={revision}")
    dataset = load_dataset(repo_id, config, split=split, revision=revision)
    missing = required - set(dataset.column_names)
    if missing:
        raise RuntimeError(f"{repo_id} schema changed; missing columns {sorted(missing)}; got {dataset.column_names}")
    if required_any and not (set(required_any) & set(dataset.column_names)):
        raise RuntimeError(f"{repo_id} schema changed; requires one of {required_any}; got {dataset.column_names}")
    if len(dataset) == 0:
        raise RuntimeError(f"{repo_id} returned an empty {split!r} split")
    path = DATA_DIR / filename
    # Stream rows to avoid duplicating very large EvalPlus/EvoEval test bodies in memory.
    atomic_write_jsonl(path, (dict(row) for row in dataset))
    return {
        "repo_id": repo_id, "config": config, "split": split, "revision": revision,
        "rows": len(dataset), "columns": list(dataset.column_names),
        "output": str(path.relative_to(ROOT)), "sha256": file_hash(path),
    }


def save_evoeval() -> list[dict]:
    return [save_dataset(
        f"evoeval/EvoEval_{subset}", f"evoeval_{subset}.jsonl",
        required={"task_id", "prompt", "canonical_solution", "entry_point"}, required_any=("test", "inputs"),
    ) for subset in ("difficult", "creative", "subtle", "tool_use", "combine")]


def save_lbpp() -> dict:
    repo_id = "CohereLabs/lbpp"
    revision = resolve_revision(repo_id)
    dataset = load_dataset(repo_id, "all", split="test", revision=revision)
    required = {"task_id", "language", "instruction", "completion", "signature", "test_setup", "test_list", "test_file"}
    missing = required - set(dataset.column_names)
    if missing:
        raise RuntimeError(f"LBPP schema changed; missing {sorted(missing)}")

    encoded_fields = ("completion", "test_setup", "test_list", "test_file")
    accepted, rejected = [], []
    for row in dataset:
        item = dict(row)
        errors = []
        for field in encoded_fields:
            decoded = decode_lbpp_value(item.get(field))
            if decoded is None:
                errors.append(field)
            else:
                item[field] = decoded
        if errors:
            rejected.append({"task_id": item.get("task_id"), "decode_failed_fields": errors})
        else:
            item["lbpp_decoded"] = True
            accepted.append(item)
    if rejected:
        atomic_write_jsonl(REJECT_DIR / "lbpp_decode_rejected.jsonl", rejected)
    if not accepted:
        raise RuntimeError("Every LBPP record failed safe decoding")
    path = DATA_DIR / "lbpp.jsonl"
    atomic_write_jsonl(path, accepted)
    return {
        "repo_id": repo_id, "config": "all", "split": "test", "revision": revision,
        "rows": len(accepted), "rejected_decode": len(rejected),
        "columns": list(dataset.column_names), "output": str(path.relative_to(ROOT)),
        "sha256": file_hash(path),
    }


def livecodebench_local_manifest() -> dict:
    """Use local LCB shards. Exact duplicates are removed later from normalized statements."""
    files = sorted(DATA_DIR.glob("livecodebench_v*.jsonl"))
    if not files:
        if os.getenv("ALLOW_MISSING_LIVECODEBENCH", "0") != "1":
            raise RuntimeError(
                "No data/coding/livecodebench_v*.jsonl files found. Add the local shards or set "
                "ALLOW_MISSING_LIVECODEBENCH=1 explicitly."
            )
        print("[Warn] LiveCodeBench explicitly omitted by ALLOW_MISSING_LIVECODEBENCH=1")
    details = []
    for path in files:
        count = 0
        first = None
        for row in iter_jsonl(path, strict=True):
            first = first or row
            count += 1
        if first is None:
            raise RuntimeError(f"LiveCodeBench shard is empty: {path}")
        required = {"question_content"}
        if not required <= set(first):
            raise RuntimeError(f"Unexpected LiveCodeBench schema in {path}: {sorted(first)}")
        details.append({"path": str(path.relative_to(ROOT)), "rows": count,
                        "sha256": file_hash(path)})
    return {"repo_id": "livecodebench/code_generation_lite", "mode": "local_shards", "files": details}


def main() -> int:
    entries: list[dict] = []
    entries.append(save_dataset("openai/openai_humaneval", "openai_humaneval.jsonl",
                                required={"task_id", "prompt", "canonical_solution", "test", "entry_point"}))
    entries.append(save_dataset("Muennighoff/mbpp", "mbpp_sanitized.jsonl", config="sanitized",
                                required={"task_id", "prompt", "code", "test_list"}))
    entries.append(save_dataset("evalplus/humanevalplus", "humanevalplus.jsonl",
                                required={"task_id", "prompt", "canonical_solution", "test", "entry_point"}))
    entries.append(save_dataset("evalplus/mbppplus", "mbppplus.jsonl",
                                required={"task_id", "prompt", "code", "test_list", "test"}))
    entries.append(save_dataset("cruxeval-org/cruxeval", "cruxeval.jsonl",
                                required={"id", "code", "input", "output"}))
    entries.append(save_dataset("FudanSELab/ClassEval", "classeval.jsonl",
                                required={"task_id", "skeleton", "solution_code", "test", "class_name"}))
    entries.extend(save_evoeval())
    entries.append(save_lbpp())
    entries.append(livecodebench_local_manifest())
    atomic_write_json(MANIFEST_DIR / "download_manifest.json", {
        "pipeline_version": PIPELINE_VERSION,
        "datasets": entries,
    })
    print(f"[Done] Wrote {MANIFEST_DIR / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
