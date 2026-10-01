from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import pickle
import zlib

import pytest

import stage1_extract
from dataset_codecs import decode_lbpp_value
from download_datasets import LIVECODEBENCH_FILES, MBPP_REPO, materialize_download
from llm_pool import Endpoint, LLMPool, file_hash, iter_jsonl
from stage1_extract import EXPECTED_BASIS, Stats, dedup_key, fmt_crux, make_sections, missing_required_kinds
from stage2_taxonomy import per_node_rng, weighted_sample_without_replacement
from stage3_question_bank import allocate_quotas, resumed_identifier_set, validate_signature


def encode_lbpp(value):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(value)))).decode()


def test_lbpp_decoder_completes_json_layer():
    code = "def fresh_name(x):\n    return x + 1\n"
    assert decode_lbpp_value(encode_lbpp(code)) == code
    tests = ["assert fresh_name(1) == 2"]
    assert decode_lbpp_value(encode_lbpp(tests)) == tests


def test_lbpp_decoder_blocks_globals_and_errors():
    malicious = base64.b64encode(zlib.compress(pickle.dumps(os.system))).decode()
    assert decode_lbpp_value(malicious) is None
    assert decode_lbpp_value("not base64") is None


def test_file_hash_streams_to_the_expected_digest(tmp_path):
    payload = (b"0123456789abcdef" * 200_000) + b"tail"
    artifact = tmp_path / "large.bin"
    artifact.write_bytes(payload)
    assert file_hash(artifact, chunk_size=4093) == hashlib.sha256(payload).hexdigest()


def test_livecodebench_release_mapping_and_materialization(tmp_path):
    assert MBPP_REPO == "google-research-datasets/mbpp"
    assert LIVECODEBENCH_FILES == {
        "test.jsonl": "livecodebench_v1.jsonl",
        "test2.jsonl": "livecodebench_v2.jsonl",
        "test3.jsonl": "livecodebench_v3.jsonl",
        "test4.jsonl": "livecodebench_v4.jsonl",
        "test5.jsonl": "livecodebench_v5.jsonl",
        "test6.jsonl": "livecodebench_v6.jsonl",
    }
    blob = tmp_path / "cache" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_text('{"question_content":"example"}\n', encoding="utf-8")
    source = tmp_path / "cache" / "snapshots" / "revision" / "test6.jsonl"
    source.parent.mkdir(parents=True)
    source.symlink_to("../../blobs/abc")
    target = tmp_path / "data" / "livecodebench_v6.jsonl"
    target.parent.mkdir()
    materialize_download(source, target)
    assert target.read_bytes() == blob.read_bytes()
    assert not target.is_symlink()
    assert not (target.parent / "livecodebench_v6.jsonl.part").exists()


def test_strict_jsonl_reader_rejects_missing_or_broken_files(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError, match="broken symlink"):
        list(iter_jsonl(missing, strict=True))
    broken = tmp_path / "broken.jsonl"
    broken.symlink_to("does-not-exist")
    with pytest.raises(FileNotFoundError, match="broken symlink"):
        list(iter_jsonl(broken, strict=True))


def test_structured_probe_rejects_schema_violations():
    pool = object.__new__(LLMPool)
    pool.mode = "json_schema"
    pool.eps = [Endpoint("http://unused", "fake-model")]

    async def fake_chat(*args, **kwargs):
        return '{"ok":"not-a-boolean"}', "stop", None

    pool.chat = fake_chat
    with pytest.raises(RuntimeError, match="violates the stage schema"):
        asyncio.run(pool._probe_structured({
            "type": "object", "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
        }))


def test_crux_sections_are_trace_not_test_or_solution():
    built = fmt_crux({"code": "def f(x): return x", "input": "1", "output": "1"}, Stats(), "cruxeval")
    sections = make_sections(built.sections)
    assert [section.kind for section in sections] == ["trace_program", "trace_input", "trace_output"]
    assert all(section.kind in EXPECTED_BASIS["trace"] for section in sections)
    assert all(section.kind not in EXPECTED_BASIS["tests"] | EXPECTED_BASIS["solution"] for section in sections)
    assert not missing_required_kinds("cruxeval", sections)
    assert missing_required_kinds("humanevalplus", sections) == {"statement", "reference_solution", "test"}


def test_declaration_only_function_validation():
    assert validate_signature("def new_metric(values: list[int]) -> int:\n    pass", "function") is None
    assert validate_signature("def new_metric(values: list[int]) -> int:\n    return sum(values)", "function")
    assert validate_signature("def a():\n    pass\ndef b():\n    pass", "function")
    assert validate_signature("def untyped(value):\n    pass", "function")
    assert validate_signature("async def async_metric(value: int) -> int:\n    pass", "function")
    assert validate_signature("def unsafe(value: int = open('x')) -> int:\n    pass", "function")
    assert validate_signature("def custom(value: UnknownType) -> int:\n    pass", "function")
    assert validate_signature("# reads stdin, writes stdout", "stdin_program") is None
    assert validate_signature("print(input())", "stdin_program")


def test_declaration_only_class_validation():
    good = """class FreshLedger:
    def __init__(self) -> None:
        pass
    def add(self, value: int) -> None:
        ...
    def total(self) -> int:
        pass
"""
    bad = good.replace("        ...", "        self.value = value")
    assert validate_signature(good, "class") is None
    assert validate_signature(bad, "class")


def test_nonoverlap_families_deduplicate_only_identical_full_material():
    assert dedup_key("cruxeval", "1", "same-code", "code-input-a") != dedup_key(
        "cruxeval", "2", "same-code", "code-input-b"
    )
    assert dedup_key("livecodebench", "x", "same-statement", "tests-a") == dedup_key(
        "livecodebench", "y", "same-statement", "tests-b"
    )


def test_rejection_compaction_keeps_only_latest_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(stage1_extract, "OUTPUT_DIR", tmp_path)
    (tmp_path / "autopsies.jsonl").write_text('{"sample_id":"accepted"}\n', encoding="utf-8")
    (tmp_path / "rejected.jsonl").write_text(
        '\n'.join([
            '{"sample_id":"open","reason":"old"}',
            '{"sample_id":"accepted","reason":"stale"}',
            '{"sample_id":"open","reason":"latest"}',
        ]) + '\n', encoding="utf-8"
    )
    stage1_extract.compact_rejections()
    rows = [json.loads(line) for line in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert rows == [{"sample_id": "open", "reason": "latest"}]


def test_taxonomy_sampling_is_deterministic_and_without_replacement():
    node = {"id": "n", "leaf_ids": ["c", "a", "b"]}
    first = weighted_sample_without_replacement(per_node_rng(node), [(str(i), i + 1) for i in range(20)], 7)
    second = weighted_sample_without_replacement(per_node_rng(node), [(str(i), i + 1) for i in range(20)], 7)
    assert first == second
    assert len({item for item, _ in first}) == 7


def test_conflicting_resume_declarations_fail_closed():
    rows = [
        {"private_design": {"signature": "def domain_counter(x: int) -> int:\n    pass"}},
        {"private_design": {"signature": "def domain_counter(y: int) -> int:\n    pass"}},
    ]
    with pytest.raises(RuntimeError, match="Conflicting declaration"):
        resumed_identifier_set(rows)


def test_weighted_quotas_are_exact_and_cover_every_cell():
    cells = {"small": ["a"], "medium": list("bcde"), "large": list("fghijklmnop")}
    quotas = allocate_quotas(cells, 100)
    assert sum(quotas.values()) == 100
    assert all(value >= 1 for value in quotas.values())
    assert quotas["large"] > quotas["medium"] > quotas["small"]
