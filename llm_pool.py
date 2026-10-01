#!/usr/bin/env python3
"""Reliable helpers shared by all LLM-backed pipeline stages."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import aiohttp
from jsonschema import SchemaError as JsonSchemaError
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema.validators import validator_for


def stable_hash(value: Any, length: int = 64) -> str:
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:length]


def file_hash(path: Path, length: int = 64, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash large artifacts without loading multi-gigabyte benchmark shards into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def validate_schema_instance(value: Any, schema: dict) -> str | None:
    """Return a concise validation error, or ``None`` when value conforms."""
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        validator_cls(schema).validate(value)
    except (JsonSchemaValidationError, JsonSchemaError) as exc:
        return exc.message
    return None


def extract_json(text: str | None):
    """Return ``(dict, error)`` from a possibly fenced model response."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    if re.search(r"<think>", text, re.I):
        return None, "unterminated_think"
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, None
        return None, "not_object"
    return None, "no_json"


def iter_jsonl(path: Path, *, strict: bool = False) -> Iterator[dict]:
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"JSONL file does not exist or is a broken symlink: {path}")
        return
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                continue
            if isinstance(obj, dict):
                yield obj
            elif strict:
                raise ValueError(f"{path}:{line_no}: expected a JSON object")


def read_jsonl(path: Path, *, strict: bool = False) -> list[dict]:
    return list(iter_jsonl(path, strict=strict))


def done_ids(path: Path, key: str) -> set[str]:
    return {str(r[key]) for r in iter_jsonl(path) if key in r}


def ensure_newline(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")


class JsonlWriter:
    """Flush-on-write append writer. Use one writer per file/process."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_newline(path)
        self.path = path
        self.fh = path.open("a", encoding="utf-8")

    def write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if not self.fh.closed:
            self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Endpoint:
    base: str
    model: str = ""
    inflight: int = 0
    consecutive_failures: int = 0


class LLMPool:
    """Least-busy OpenAI-compatible multi-server client with retries and schema output."""

    def __init__(
        self,
        servers: list[str],
        session: aiohttp.ClientSession,
        *,
        concurrency: int,
        timeout: float = 300,
        max_retries: int = 5,
        temperature: float = 0.2,
        mode: str = "json_schema",
        extra_body: Optional[str | dict] = None,
        served_model_id: Optional[str] = None,
        allow_mixed_models: bool = False,
        errors: Optional[Counter] = None,
        error_bodies: Optional[list[str]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ):
        self.eps = [Endpoint(x.rstrip("/")) for x in servers]
        self.session = session
        self.sem = semaphore or asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.mode = mode
        self.served_model_id = served_model_id
        self.allow_mixed_models = allow_mixed_models
        self.extra = json.loads(extra_body) if isinstance(extra_body, str) else dict(extra_body or {})
        self.errors = errors if errors is not None else Counter()
        self.error_bodies = error_bodies if error_bodies is not None else []

    @property
    def model_ids(self) -> list[str]:
        return sorted({ep.model for ep in self.eps if ep.model})

    @property
    def model_signature(self) -> str:
        return ",".join(self.model_ids)

    def _log_body(self, reason: str, body: str) -> None:
        self.errors[reason] += 1
        if len(self.error_bodies) < 10:
            self.error_bodies.append(f"{reason}: {body[:500]}")

    async def startup(self, probe_schema: Optional[dict] = None) -> None:
        async def inspect(ep: Endpoint):
            try:
                timeout = aiohttp.ClientTimeout(total=20)
                async with self.session.get(f"{ep.base}/models", timeout=timeout) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:200]}")
                    payload = await response.json()
                ids = [str(item["id"]) for item in payload.get("data", []) if item.get("id")]
                if not ids:
                    raise RuntimeError("empty model list")
                if self.served_model_id:
                    if self.served_model_id not in ids:
                        raise RuntimeError(f"requested {self.served_model_id!r}, serves {ids}")
                    ep.model = self.served_model_id
                else:
                    ep.model = ids[0]
                print(f"  [Endpoint] {ep.base} ONLINE model={ep.model!r}")
                return ep
            except Exception as exc:
                print(f"  [Endpoint] {ep.base} UNREACHABLE: {type(exc).__name__}: {exc}")
                return None

        self.eps = [x for x in await asyncio.gather(*(inspect(ep) for ep in self.eps)) if x]
        if not self.eps:
            raise RuntimeError("No reachable LLM endpoint")
        if len(self.model_ids) > 1 and not self.allow_mixed_models:
            raise RuntimeError(f"Endpoints serve different models: {self.model_ids}. Set ALLOW_MIXED_MODELS=1 explicitly to allow this.")
        await self._probe_structured(probe_schema)

    async def _probe_structured(self, real_schema: Optional[dict]) -> None:
        if self.mode == "none":
            return
        schema = real_schema or {
            "type": "object",
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        # Probe schema compilation with a tiny request. A real stage schema catches backend/compiler
        # incompatibilities before thousands of requests are queued.
        prompt = "Return one minimal valid JSON object for the supplied schema."
        text, _, err = await self.chat(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "go"}],
            2048,
            schema=schema,
            schema_name="startup_probe",
            _skip_probe=True,
        )
        obj, parse_error = extract_json(text) if not err else (None, err)
        if obj is None:
            raise RuntimeError(f"Structured-output mode {self.mode!r} failed the stage schema probe: {parse_error}")
        try:
            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            validator_cls(schema).validate(obj)
        except (JsonSchemaValidationError, JsonSchemaError) as exc:
            raise RuntimeError(
                f"Structured-output mode {self.mode!r} returned JSON that violates the stage schema: {exc.message}"
            ) from exc
        print(f"  [Structured] mode={self.mode!r} schema compilation and enforcement verified")

    def _pick(self) -> Endpoint:
        healthy = [ep for ep in self.eps if ep.consecutive_failures < 5]
        if not healthy:
            for ep in self.eps:
                ep.consecutive_failures = 0
            healthy = self.eps
        return min(healthy, key=lambda ep: (ep.inflight, ep.base))

    def _request_body(
        self,
        ep: Endpoint,
        messages: list,
        max_tokens: int,
        schema: Optional[dict],
        schema_name: str,
        temperature: Optional[float],
    ) -> dict:
        body: dict[str, Any] = {
            "model": ep.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if schema is not None:
            if self.mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                }
            elif self.mode == "guided_json":
                body["guided_json"] = schema
            elif self.mode == "structured_outputs":
                body["structured_outputs"] = {"json": schema}
        reserved = {"model", "messages", "max_tokens", "temperature", "response_format", "guided_json", "structured_outputs"}
        collision = reserved & self.extra.keys()
        if collision:
            raise ValueError(f"EXTRA_REQUEST_BODY may not override {sorted(collision)}")
        body.update(self.extra)
        return body

    async def chat(
        self,
        messages: list,
        max_tokens: int,
        *,
        schema: Optional[dict] = None,
        schema_name: str = "output",
        temperature: Optional[float] = None,
        _skip_probe: bool = False,
    ):
        last_error = "exhausted"
        for attempt in range(self.max_retries):
            ep = self._pick()
            ep.inflight += 1
            status: Optional[int] = None
            response_body = ""
            try:
                async with self.sem:
                    async with self.session.post(
                        f"{ep.base}/chat/completions",
                        json=self._request_body(ep, messages, max_tokens, schema, schema_name, temperature),
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        status = response.status
                        response_body = await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"connection:{type(exc).__name__}"
                ep.consecutive_failures += 1
                self.errors[last_error] += 1
            finally:
                ep.inflight -= 1

            if status == 200:
                try:
                    choice = json.loads(response_body)["choices"][0]
                    ep.consecutive_failures = 0
                    return choice["message"].get("content") or "", choice.get("finish_reason"), None
                except Exception:
                    last_error = "bad_response_shape"
                    self._log_body(last_error, response_body)
            elif status is not None:
                last_error = f"http_{status}"
                if status in {408, 425, 429} or status >= 500:
                    self.errors[last_error] += 1
                    if status >= 500:
                        ep.consecutive_failures += 1
                else:
                    self._log_body(last_error, response_body)
                    return None, None, last_error
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
        return None, None, last_error
