from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import pandas as pd

RENAME_MAP = {
    "名称": "name",
    "交易日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(手)": "volume",
    "涨跌幅(%)": "pct_chg",
    "前收盘价": "prev_close",
    "所属行业": "industry",
    "地域": "region",
}

NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "pct_chg", "prev_close")
TEXT_COLUMNS = ("name", "industry", "region")
COLUMNAR_DIRNAME = ".columnar_store"
PARQUET_DIRNAME = "parquet"
DB_FILENAME = "market_data.duckdb"
POOL_WORKERS = max(1, min(os.cpu_count() or 1, 8))
MAP_CHUNK_SIZE = 8


def columnar_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / COLUMNAR_DIRNAME


def parquet_root(data_dir: str | Path) -> Path:
    return columnar_root(data_dir) / PARQUET_DIRNAME


def database_path(data_dir: str | Path) -> Path:
    return columnar_root(data_dir) / DB_FILENAME


def parquet_path_for(csv_path: str | Path, data_dir: str | Path) -> Path:
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    return parquet_root(data_dir) / csv_path.relative_to(data_dir).with_suffix(
        ".parquet"
    )


def _normalize_frame(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    if df.empty:
        return df

    if source_type == "csv":
        df.rename(columns=RENAME_MAP, inplace=True)

    if "date" not in df.columns:
        return pd.DataFrame()

    if source_type == "csv":
        raw_date = pd.to_numeric(df["date"], errors="coerce")
        df = df[raw_date.notna()].copy()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(
            raw_date.loc[df.index].astype("int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.dropna(subset=["date"], inplace=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _convert_single_stock(args: tuple[str, str]) -> dict | None:
    csv_path, data_dir = args
    csv_file = Path(csv_path)
    try:
        df = pd.read_csv(csv_file, usecols=lambda c: c in RENAME_MAP)
    except Exception:
        return None
    df = _normalize_frame(df, "csv")
    if df.empty:
        return None

    out_path = parquet_path_for(csv_file, data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    stat = csv_file.stat()
    return {
        "code": csv_file.stem[:6],
        "csv_path": str(csv_file),
        "parquet_path": str(out_path),
        "rows": int(len(df)),
        "first_date": df["date"].iloc[0].date().isoformat(),
        "last_date": df["date"].iloc[-1].date().isoformat(),
        "name": str(df["name"].iloc[-1]) if "name" in df.columns else csv_file.stem[:6],
        "industry": str(df["industry"].iloc[-1]) if "industry" in df.columns else "",
        "region": str(df["region"].iloc[-1]) if "region" in df.columns else "",
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_size": int(stat.st_size),
    }


def _load_existing_index(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[int, int]]:
    try:
        rows = con.execute(
            "SELECT csv_path, source_mtime_ns, source_size FROM stock_files"
        ).fetchall()
    except duckdb.Error:
        return {}
    return {row[0]: (int(row[1]), int(row[2])) for row in rows}


def sync_parquet_store(data_dir: str, workers: int | None = None) -> dict:
    data_dir_path = Path(data_dir)
    root = columnar_root(data_dir_path)
    root.mkdir(parents=True, exist_ok=True)

    csv_files = []
    for root_dir, _, files in os.walk(data_dir):
        if (
            COLUMNAR_DIRNAME in root_dir
            or ".frame_cache" in root_dir
            or "大盘" in root_dir
        ):
            continue
        for name in files:
            if name.endswith(".csv") and name[:6].isdigit():
                csv_files.append(str(Path(root_dir) / name))
    csv_files.sort()

    db_path = database_path(data_dir_path)
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_files (
            csv_path TEXT PRIMARY KEY,
            code TEXT,
            parquet_path TEXT,
            rows BIGINT,
            first_date DATE,
            last_date DATE,
            name TEXT,
            industry TEXT,
            region TEXT,
            source_mtime_ns BIGINT,
            source_size BIGINT,
            synced_at TIMESTAMP DEFAULT now()
        )
        """
    )
    existing = _load_existing_index(con)
    current_paths = set(csv_files)
    existing_paths = set(existing)

    removed = sorted(existing_paths - current_paths)
    if removed:
        con.executemany(
            "DELETE FROM stock_files WHERE csv_path = ?", [(p,) for p in removed]
        )

    tasks = []
    unchanged = 0
    for csv_path in csv_files:
        stat = Path(csv_path).stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        if (
            existing.get(csv_path) == signature
            and parquet_path_for(csv_path, data_dir_path).exists()
        ):
            unchanged += 1
            continue
        tasks.append((csv_path, str(data_dir_path)))

    records = []
    if tasks:
        with ProcessPoolExecutor(max_workers=workers or POOL_WORKERS) as executor:
            for record in executor.map(
                _convert_single_stock, tasks, chunksize=MAP_CHUNK_SIZE
            ):
                if record:
                    records.append(record)
        if records:
            con.executemany(
                """
                INSERT OR REPLACE INTO stock_files (
                    csv_path, code, parquet_path, rows, first_date, last_date, name, industry, region,
                    source_mtime_ns, source_size, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
                """,
                [
                    (
                        r["csv_path"],
                        r["code"],
                        r["parquet_path"],
                        r["rows"],
                        r["first_date"],
                        r["last_date"],
                        r["name"],
                        r["industry"],
                        r["region"],
                        r["source_mtime_ns"],
                        r["source_size"],
                    )
                    for r in records
                ],
            )

    con.execute("CREATE INDEX IF NOT EXISTS idx_stock_files_code ON stock_files(code)")
    total = con.execute("SELECT COUNT(*) FROM stock_files").fetchone()[0]
    con.close()

    return {
        "total": int(total),
        "converted": len(records),
        "unchanged": int(unchanged),
        "removed": len(removed),
        "db_path": str(db_path),
        "parquet_root": str(parquet_root(data_dir_path)),
    }


def resolve_stock_files(data_dir: str, prefer_parquet: bool = True) -> list[str]:
    db_path = database_path(data_dir)
    if not db_path.exists():
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT csv_path, parquet_path FROM stock_files ORDER BY code"
    ).fetchall()
    con.close()

    selected = []
    for csv_path, parquet_path in rows:
        csv_file = Path(csv_path)
        parquet_file = Path(parquet_path)
        if prefer_parquet and parquet_file.exists():
            try:
                if (
                    not csv_file.exists()
                ) or parquet_file.stat().st_mtime_ns >= csv_file.stat().st_mtime_ns:
                    selected.append(str(parquet_file))
                    continue
            except FileNotFoundError:
                selected.append(str(parquet_file))
                continue
        if csv_file.exists():
            selected.append(str(csv_file))
    return selected


def get_stock_source_path(
    data_dir: str, stock_code: str, prefer_parquet: bool = True
) -> str | None:
    db_path = database_path(data_dir)
    if db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        row = con.execute(
            "SELECT csv_path, parquet_path FROM stock_files WHERE code = ? LIMIT 1",
            [stock_code],
        ).fetchone()
        con.close()
        if row:
            csv_path, parquet_path = row
            csv_file = Path(csv_path)
            parquet_file = Path(parquet_path)
            if prefer_parquet and parquet_file.exists():
                try:
                    if (
                        not csv_file.exists()
                    ) or parquet_file.stat().st_mtime_ns >= csv_file.stat().st_mtime_ns:
                        return str(parquet_file)
                except FileNotFoundError:
                    return str(parquet_file)
            if csv_file.exists():
                return str(csv_file)

    for root_dir, _, files in os.walk(data_dir):
        if COLUMNAR_DIRNAME in root_dir or ".frame_cache" in root_dir:
            continue
        for name in files:
            if name == f"{stock_code}.csv":
                return str(Path(root_dir) / name)
    return None
