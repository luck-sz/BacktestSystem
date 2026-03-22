from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from columnar_store import resolve_stock_files

FACTOR_DIRNAME = ".factor_store"
FACTOR_SUBDIR = "base"
DB_FILENAME = "factor_store.duckdb"
POOL_WORKERS = max(1, min(os.cpu_count() or 1, 8))
MAP_CHUNK_SIZE = 8


def factor_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / FACTOR_DIRNAME


def factor_data_root(data_dir: str | Path) -> Path:
    return factor_root(data_dir) / FACTOR_SUBDIR


def factor_db_path(data_dir: str | Path) -> Path:
    return factor_root(data_dir) / DB_FILENAME


def factor_path_for(source_path: str | Path, data_dir: str | Path) -> Path:
    source = Path(source_path)
    data_dir = Path(data_dir)
    if ".columnar_store" in source.parts:
        idx = source.parts.index("parquet")
        rel = Path(*source.parts[idx + 1 :]).with_suffix(".parquet")
    else:
        rel = source.relative_to(data_dir).with_suffix(".parquet")
    return factor_data_root(data_dir) / rel


def _compute_base_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": df["date"]})

    low21 = df["low"].rolling(21, min_periods=1).min()
    high21 = df["close"].rolling(21, min_periods=1).max()
    out["rsv_long"] = ((df["close"] - low21) / (high21 - low21 + 1e-9) * 100).astype(
        "float32"
    )

    low3 = df["low"].rolling(3, min_periods=1).min()
    high3 = df["close"].rolling(3, min_periods=1).max()
    out["rsv_short"] = ((df["close"] - low3) / (high3 - low3 + 1e-9) * 100).astype(
        "float32"
    )

    out["vol_ratio"] = (df["volume"] / df["volume"].shift(1).replace(0, np.nan)).astype(
        "float32"
    )
    out["turnover"] = (df["close"] * df["volume"]).astype("float32")
    out["avg_turnover20"] = (
        out["turnover"].rolling(20, min_periods=1).mean().astype("float32")
    )
    out["max_vol120"] = df["volume"].rolling(120, min_periods=1).max().astype("float32")
    out["prev_close_calc"] = df["close"].shift(1).astype("float32")

    out["ma5"] = df["close"].rolling(5).mean().astype("float32")
    out["ma60"] = df["close"].rolling(60).mean().astype("float32")

    low9 = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    diff9 = (high9 - low9).clip(lower=0.001)
    rsv = (df["close"] - low9) / diff9 * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    out["j"] = (3 * k - 2 * d).astype("float32")

    hhv4 = df["high"].rolling(4).max()
    llv4 = df["low"].rolling(4).min()
    diff4 = (hhv4 - llv4).clip(lower=0.001)

    var1a = (hhv4 - df["close"]) / diff4 * 100 - 90
    var2a = var1a.ewm(alpha=1 / 4, adjust=False).mean() + 100
    var3a = (df["close"] - llv4) / diff4 * 100
    var4a = var3a.ewm(alpha=1 / 6, adjust=False).mean()
    var5a = var4a.ewm(alpha=1 / 6, adjust=False).mean() + 100
    v6a = (var5a - var2a).fillna(0)
    brick = np.where(v6a > 4, v6a - 4, 0.0)
    out["brick"] = pd.Series(brick, index=df.index).astype("float32")

    b0 = out["brick"]
    b1 = b0.shift(1)
    b2 = b0.shift(2)
    red_len = b0 - b1
    green_len = b2 - b1
    color_ok = (b1 < b2) & (b0 > b1)
    power_ok = red_len > green_len
    oversold_ok = out["j"].shift(1).rolling(5).min() < 0
    below_ma5_5days = (df["close"] < out["ma5"]).shift(1).rolling(5).sum() == 5
    is_breakout = df["close"] > out["ma5"]

    out["brick_score_pre"] = (red_len / (green_len + 0.001)).astype("float32")
    out["brick_buy_signal"] = (
        color_ok & power_ok & oversold_ok & below_ma5_5days & is_breakout
    ).fillna(False)

    m0 = out["brick"]
    m1 = m0.shift(1)
    m2 = m0.shift(2)
    drop_len_0 = m1 - m0
    drop_len_1 = m2 - m1
    is_falling = drop_len_0 > 0
    is_decelerating = drop_len_0 < drop_len_1
    j_ok = (out["j"] > out["j"].shift(1)) & (out["j"].shift(1) < 0)
    trend_ok = df["close"] > out["ma60"]
    out["green_buy_signal"] = (is_falling & is_decelerating & j_ok & trend_ok).fillna(
        False
    )
    out["green_score"] = (
        (1.0 / (drop_len_0 + 1e-6)).replace([np.inf, -np.inf], np.nan).astype("float32")
    )

    return out


def _build_single_factor(args: tuple[str, str]) -> dict | None:
    source_path, data_dir = args
    source = Path(source_path)
    try:
        if source.suffix.lower() == ".parquet":
            df = pd.read_parquet(source)
        else:
            return None
    except Exception:
        return None

    if df.empty or "date" not in df.columns:
        return None

    factors = _compute_base_factors(df)
    out_path = factor_path_for(source, data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(out_path, index=False)

    stat = source.stat()
    return {
        "source_path": str(source),
        "factor_path": str(out_path),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
        "rows": int(len(factors)),
    }


def sync_factor_store(data_dir: str, workers: int | None = None) -> dict:
    root = factor_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    db_path = factor_db_path(data_dir)
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_files (
            source_path TEXT PRIMARY KEY,
            factor_path TEXT,
            source_mtime_ns BIGINT,
            source_size BIGINT,
            rows BIGINT,
            synced_at TIMESTAMP DEFAULT now()
        )
        """
    )

    existing = {}
    try:
        for row in con.execute(
            "SELECT source_path, source_mtime_ns, source_size FROM factor_files"
        ).fetchall():
            existing[row[0]] = (int(row[1]), int(row[2]))
    except duckdb.Error:
        pass

    sources = resolve_stock_files(data_dir, prefer_parquet=True)
    tasks = []
    unchanged = 0
    for source_path in sources:
        source = Path(source_path)
        if source.suffix.lower() != ".parquet":
            continue
        stat = source.stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        target = factor_path_for(source, data_dir)
        if existing.get(str(source)) == signature and target.exists():
            unchanged += 1
            continue
        tasks.append((str(source), data_dir))

    built = []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers or POOL_WORKERS) as executor:
            for row in executor.map(
                _build_single_factor, tasks, chunksize=MAP_CHUNK_SIZE
            ):
                if row:
                    built.append(row)
        if built:
            con.executemany(
                """
                INSERT OR REPLACE INTO factor_files (
                    source_path, factor_path, source_mtime_ns, source_size, rows, synced_at
                ) VALUES (?, ?, ?, ?, ?, now())
                """,
                [
                    (
                        r["source_path"],
                        r["factor_path"],
                        r["source_mtime_ns"],
                        r["source_size"],
                        r["rows"],
                    )
                    for r in built
                ],
            )

    total = con.execute("SELECT COUNT(*) FROM factor_files").fetchone()[0]
    con.close()
    return {
        "total": int(total),
        "converted": len(built),
        "unchanged": int(unchanged),
        "db_path": str(db_path),
    }


def get_factor_path(source_path: str, data_dir: str) -> str | None:
    target = factor_path_for(source_path, data_dir)
    return str(target) if target.exists() else None
