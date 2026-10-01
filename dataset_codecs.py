"""Safe decoders for encoded benchmark fields."""
from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from typing import Any


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def decode_lbpp_value(value: Any) -> Any | None:
    """Decode LBPP's base64 -> zlib -> pickle -> JSON representation safely."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        compressed = base64.b64decode(value.encode("utf-8"), validate=True)
        raw = zlib.decompress(compressed)
        serialized_json = RestrictedUnpickler(io.BytesIO(raw)).load()
        if not isinstance(serialized_json, (str, bytes, bytearray)):
            return None
        decoded = json.loads(serialized_json)
        return decoded if isinstance(decoded, (str, list, dict)) else None
    except Exception:
        return None
