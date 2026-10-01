"""Shared, environment-overridable configuration for the data-generation pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "coding"
AUTOPSY_DIR = ROOT / "output" / "autopsies"
TAXONOMY_DIR = ROOT / "output" / "taxonomy"
QUESTION_DIR = ROOT / "output" / "question_bank"

# Requested global limit. Each stage also sizes its aiohttp connector above this value.
GLOBAL_CONCURRENCY = 300
HTTP_REQUEST_TIMEOUT = float(os.getenv("HTTP_REQUEST_TIMEOUT", "300"))
MAX_HTTP_RETRIES = int(os.getenv("MAX_HTTP_RETRIES", "5"))


def _servers(name: str, default: str) -> list[str]:
    return [x.strip().rstrip("/") for x in os.getenv(name, default).split(",") if x.strip()]


SERVERS = _servers(
    "VLLM_SERVERS",
    "http://cn23-a40:8000/v1,http://cn24-a40:8001/v1,http://cn24-a40:8002/v1",
)
# Use a genuinely different model here when possible. It intentionally has a separate setting.
JUDGE_SERVERS = _servers("JUDGE_VLLM_SERVERS", ",".join(SERVERS))
SERVED_MODEL_ID = os.getenv("SERVED_MODEL_ID") or None
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID") or None
STRUCTURED_OUTPUT_MODE = os.getenv("STRUCTURED_OUTPUT_MODE", "json_schema")
ALLOW_MIXED_MODELS = os.getenv("ALLOW_MIXED_MODELS", "0") == "1"
# Set to 1 for the strongest quality gate. It is optional by default so a single-model
# local setup remains usable, but every report records whether review was independent.
REQUIRE_DISTINCT_JUDGE_MODEL = os.getenv("REQUIRE_DISTINCT_JUDGE_MODEL", "0") == "1"

# Disabling hidden chain-of-thought avoids spending output budgets on reasoning tokens. Set this
# environment variable to an empty JSON object if a server/chat template rejects the kwarg.
_EXTRA = os.getenv("EXTRA_REQUEST_BODY", '{"chat_template_kwargs":{"enable_thinking":false}}')
EXTRA_REQUEST_BODY: dict[str, Any] = json.loads(_EXTRA) if _EXTRA.strip() else {}

PIPELINE_VERSION = "2026-10-01.5"
