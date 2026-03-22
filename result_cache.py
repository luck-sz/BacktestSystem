from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_DIRNAME = ".result_cache"


def cache_root(base_dir: str | Path) -> Path:
    return Path(base_dir) / CACHE_DIRNAME


def _kind_dir(base_dir: str | Path, kind: str) -> Path:
    return cache_root(base_dir) / kind


def _payload_path(base_dir: str | Path, kind: str, cache_key: str) -> Path:
    return _kind_dir(base_dir, kind) / f"{cache_key}.json.gz"


def _signature_for(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def build_data_signature(base_dir: str | Path, data_dir: str | Path) -> dict[str, Any]:
    base = Path(base_dir)
    data = Path(data_dir)
    signature = {
        "columnar_db": _signature_for(data / ".columnar_store" / "market_data.duckdb"),
        "factor_db": _signature_for(data / ".factor_store" / "factor_store.duckdb"),
        "update_lock": None,
    }
    lock_path = data / ".update_lock"
    if lock_path.exists():
        try:
            signature["update_lock"] = lock_path.read_text(encoding="utf-8").strip()
        except Exception:
            signature["update_lock"] = None
    return signature


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def generic_cache_key(
    kind: str, base_dir: str | Path, data_dir: str | Path, params: dict[str, Any]
) -> str:
    payload = {
        "kind": kind,
        "params": params,
        "data_signature": build_data_signature(base_dir, data_dir),
    }
    return _stable_hash(payload)


def backtest_cache_key(
    base_dir: str | Path, data_dir: str | Path, params: dict[str, Any]
) -> str:
    return generic_cache_key("backtest", base_dir, data_dir, params)


def selection_cache_key(
    base_dir: str | Path, data_dir: str | Path, params: dict[str, Any]
) -> str:
    return generic_cache_key("selection", base_dir, data_dir, params)


def load_cached_payload(
    base_dir: str | Path, kind: str, cache_key: str
) -> dict[str, Any] | None:
    path = _payload_path(base_dir, kind, cache_key)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def save_cached_payload(
    base_dir: str | Path, kind: str, cache_key: str, payload: dict[str, Any]
) -> str:
    out = _payload_path(base_dir, kind, cache_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return str(out)
