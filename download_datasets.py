#!/usr/bin/env python3
"""Download and normalize the benchmark sources used by the pipeline.

Dataset revisions are resolved to immutable Hugging Face commit SHAs and recorded in
``data/manifests/download_manifest.json``. LBPP's encoded code fields are decoded with a
restricted unpickler; undecodable records are rejected rather than converted into fake solutions.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from dataset_codecs import decode_lbpp_value
from llm_pool import atomic_write_json, atomic_write_jsonl, file_hash, iter_jsonl
from pipeline_config import DATA_DIR, PIPELINE_VERSION, ROOT

MANIFEST_DIR = ROOT / "data" / "manifests"
REJECT_DIR = ROOT / "data" / "rejected"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
REJECT_DIR.mkdir(parents=True, exist_ok=True)

# The repository names its first release file ``test.jsonl`` and subsequent files
# ``test2.jsonl`` through ``test6.jsonl``. Keep explicit local names so routing and
# provenance remain unambiguous in later stages.
LIVECODEBENCH_FILES = {
    "test.jsonl": "livecodebench_v1.jsonl",
    "test2.jsonl": "livecodebench_v2.jsonl",
    "test3.jsonl": "livecodebench_v3.jsonl",
    "test4.jsonl": "livecodebench_v4.jsonl",
    "test5.jsonl": "livecodebench_v5.jsonl",
    "test6.jsonl": "livecodebench_v6.jsonl",
}


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


def materialize_download(source: Path, target: Path) -> None:
    """Atomically expose a Hub-cached file under its pipeline name without needless copies."""
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        # The Hub cache is normally on the same filesystem, so a hard link avoids
        # temporarily doubling the roughly 4.5 GB LiveCodeBench footprint.
        os.link(source, temporary)
    except OSError:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    os.replace(temporary, target)


def download_livecodebench() -> dict:
    """Download and rename all six release files from the pinned Hub revision."""
    repo_id = "livecodebench/code_generation_lite"
    revision = resolve_revision(repo_id)
    details = []
    for remote_name, local_name in LIVECODEBENCH_FILES.items():
        print(f"[Download] {repo_id}/{remote_name} revision={revision} -> {local_name}")
        cached = Path(hf_hub_download(
            repo_id=repo_id, filename=remote_name, repo_type="dataset", revision=revision,
        ))
        path = DATA_DIR / local_name
        materialize_download(cached, path)

        count = 0
        required = {"question_content"}
        for line_number, row in enumerate(iter_jsonl(path, strict=True), 1):
            missing = required - set(row)
            if missing:
                raise RuntimeError(
                    f"Unexpected LiveCodeBench schema in {remote_name} line {line_number}: "
                    f"missing {sorted(missing)}; got {sorted(row)}"
                )
            count += 1
        if count == 0:
            raise RuntimeError(f"LiveCodeBench file is empty: {remote_name}")
        details.append({
            "remote_file": remote_name, "path": str(path.relative_to(ROOT)),
            "rows": count, "sha256": file_hash(path),
        })
    return {"repo_id": repo_id, "revision": revision, "mode": "hub_release_files", "files": details}


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
    entries.append(download_livecodebench())
    atomic_write_json(MANIFEST_DIR / "download_manifest.json", {
        "pipeline_version": PIPELINE_VERSION,
        "datasets": entries,
    })
    print(f"[Done] Wrote {MANIFEST_DIR / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
