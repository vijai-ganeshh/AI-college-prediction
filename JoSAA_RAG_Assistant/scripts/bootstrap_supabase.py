#!/usr/bin/env python3
"""
Bootstrap the Supabase/Postgres schema used by the JoSAA Streamlit app.

What this script does:
1. Loads DB credentials from environment or .env
2. Creates the notebook-designed core tables/views
3. Downloads JoSAA 2024 round data from the original GitHub dataset
4. Loads local NIRF CSV data from data/clean/nirf_2025_overall.csv
5. Rebuilds institute mapping and a combined JoSAA+NIRF view

The current Streamlit app only requires `josaa_btech_2024`, but this script
also restores the broader notebook-oriented schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import psycopg2
from psycopg2.extras import execute_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NIRF_CSV = ROOT / "data" / "clean" / "nirf_2025_overall.csv"
DEFAULT_ALIAS_CSV = ROOT / "data" / "clean" / "alias_map.csv"
JOSAA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/sickboydroid/JoSAA-DataSet/main/2024/round{round_num}.json"
)

MANUAL_MAPPINGS = [
    ("Indian Institute of Technology Madras", "Indian Institute of Technology Madras", 1.0),
    ("Indian Institute  of Technology Madras", "Indian Institute of Technology Madras", 1.0),
    ("Indian Institute of Technology Delhi", "Indian Institute of Technology Delhi", 1.0),
    ("Indian Institute  of Technology Delhi", "Indian Institute of Technology Delhi", 1.0),
    ("Indian Institute of Technology Bombay", "Indian Institute of Technology Bombay", 1.0),
    ("Indian Institute  of Technology Bombay", "Indian Institute of Technology Bombay", 1.0),
    ("Indian Institute of Technology Kanpur", "Indian Institute of Technology Kanpur", 1.0),
    ("Indian Institute  of Technology Kanpur", "Indian Institute of Technology Kanpur", 1.0),
    ("Indian Institute of Technology Roorkee", "Indian Institute of Technology Roorkee", 1.0),
    ("Indian Institute  of Technology Roorkee", "Indian Institute of Technology Roorkee", 1.0),
    ("Indian Institute of Technology Kharagpur", "Indian Institute of Technology Kharagpur", 1.0),
    ("Indian Institute  of Technology Kharagpur", "Indian Institute of Technology Kharagpur", 1.0),
    ("Indian Institute of Technology Guwahati", "Indian Institute of Technology Guwahati", 1.0),
    ("Indian Institute  of Technology Guwahati", "Indian Institute of Technology Guwahati", 1.0),
    ("Indian Institute of Technology Banaras Hindu University", "Indian Institute of Technology (BHU) Varanasi", 0.95),
    ("Indian Institute of Science", "Indian Institute of Science", 1.0),
    ("Jawaharlal Nehru University", "Jawaharlal Nehru University", 1.0),
    ("All India Institute of Medical Sciences, Delhi", "All India Institute of Medical Sciences, Delhi", 1.0),
]


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def db_params() -> dict:
    load_local_env(ROOT / ".env")
    required = {
        "host": os.getenv("DB_HOST", "").strip(),
        "database": os.getenv("DB_NAME", "").strip() or "postgres",
        "user": os.getenv("DB_USER", "").strip(),
        "password": os.getenv("DB_PASSWORD", "").strip(),
        "port": int(os.getenv("DB_PORT", "6543")),
        "sslmode": os.getenv("DB_SSLMODE", "require").strip() or "require",
        "connect_timeout": 15,
    }
    missing = [k.upper() for k, v in required.items() if not v and k != "connect_timeout"]
    if missing:
        raise SystemExit(f"Missing DB env vars: {', '.join(missing)}")
    return required


def normalize_spaces(value: str) -> str:
    return " ".join((value or "").strip().split())


def classify_institute_type(institute_name: str) -> str:
    name = (institute_name or "").upper()
    if "INDIAN INSTITUTE OF TECHNOLOGY" in name or name.startswith("IIT "):
        return "IIT"
    if "NATIONAL INSTITUTE OF TECHNOLOGY" in name or name.startswith("NIT "):
        return "NIT"
    if "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY" in name or name.startswith("IIIT "):
        return "IIIT"
    return "GFTI"


def fetch_json(url: str) -> list | None:
    try:
        with urlrequest.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def iter_josaa_rows() -> Iterable[Tuple[int, Tuple[int, int, str, str, str, str, str, str, str, str]]]:
    for round_num in range(1, 7):
        url = JOSAA_URL_TEMPLATE.format(round_num=round_num)
        print(f"Downloading JoSAA round {round_num}: {url}")
        records = fetch_json(url)
        if records is None:
            print(f"  round {round_num} not found; stopping round import")
            break
        print(f"  fetched {len(records):,} rows")
        for record in records:
            if len(record) != 7:
                continue
            institute = normalize_spaces(str(record[0]))
            yield round_num, (
                2024,
                round_num,
                institute,
                classify_institute_type(institute),
                normalize_spaces(str(record[1])),
                normalize_spaces(str(record[2])),
                normalize_spaces(str(record[3])),
                normalize_spaces(str(record[4])),
                str(record[5]).strip(),
                str(record[6]).strip(),
            )


def detect_nirf_year(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        first = next(reader, None)
    if not first or not first.get("year"):
        raise SystemExit(f"Could not detect year from {csv_path}")
    return int(first["year"])


def nirf_table_name(csv_path: Path) -> str:
    return f"nirf_rankings_{detect_nirf_year(csv_path)}"


def ensure_schema(conn, nirf_table: str) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS josaa_btech_2024 (
        id BIGSERIAL PRIMARY KEY,
        year INTEGER NOT NULL,
        round INTEGER NOT NULL,
        institute TEXT NOT NULL,
        institute_type TEXT NOT NULL,
        program TEXT NOT NULL,
        quota TEXT NOT NULL,
        category TEXT NOT NULL,
        gender TEXT NOT NULL,
        opening_rank TEXT,
        closing_rank TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS {nirf_table} (
        id BIGSERIAL PRIMARY KEY,
        year INTEGER NOT NULL,
        category TEXT NOT NULL,
        rank INTEGER NOT NULL,
        institute TEXT NOT NULL,
        institute_normalized TEXT,
        institute_id TEXT,
        state TEXT,
        city TEXT,
        score REAL,
        page_url TEXT,
        row_anchor TEXT,
        crawl_time TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS institute_mapping (
        id BIGSERIAL PRIMARY KEY,
        josaa_name TEXT NOT NULL UNIQUE,
        nirf_name TEXT NOT NULL,
        confidence_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_institute ON josaa_btech_2024(institute);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_institute_type ON josaa_btech_2024(institute_type);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_round ON josaa_btech_2024(round);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_category ON josaa_btech_2024(category);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_quota ON josaa_btech_2024(quota);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_gender ON josaa_btech_2024(gender);
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_program_fts ON josaa_btech_2024 USING gin(to_tsvector('english', program));
    CREATE INDEX IF NOT EXISTS idx_josaa_btech_2024_closing_rank_numeric
        ON josaa_btech_2024 ((CAST(closing_rank AS INTEGER)))
        WHERE closing_rank ~ '^[0-9]+$';

    CREATE INDEX IF NOT EXISTS idx_{nirf_table}_rank ON {nirf_table}(rank);
    CREATE INDEX IF NOT EXISTS idx_{nirf_table}_category ON {nirf_table}(category);
    CREATE INDEX IF NOT EXISTS idx_{nirf_table}_institute ON {nirf_table}(institute);
    CREATE INDEX IF NOT EXISTS idx_{nirf_table}_institute_norm ON {nirf_table}(institute_normalized);
    CREATE INDEX IF NOT EXISTS idx_{nirf_table}_state ON {nirf_table}(state);

    CREATE OR REPLACE VIEW josaa_2024 AS
    SELECT * FROM josaa_btech_2024;

    CREATE OR REPLACE VIEW nirf_rankings_latest AS
    SELECT * FROM {nirf_table};

    CREATE OR REPLACE VIEW josaa_nirf_combined AS
    SELECT
        j.id,
        j.year,
        j.round,
        j.institute,
        j.institute_type,
        j.program,
        j.quota,
        j.category,
        j.gender,
        j.opening_rank,
        j.closing_rank,
        j.created_at,
        n.rank AS nirf_rank,
        n.score AS nirf_score,
        n.category AS nirf_category,
        n.state AS nirf_state,
        n.city AS nirf_city
    FROM josaa_btech_2024 j
    LEFT JOIN institute_mapping im
        ON im.josaa_name = j.institute
    LEFT JOIN nirf_rankings_latest n
        ON regexp_replace(lower(trim(n.institute)), '\\s+', ' ', 'g') =
           regexp_replace(lower(trim(COALESCE(im.nirf_name, j.institute))), '\\s+', ' ', 'g');
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def ensure_empty_or_reset(conn, table_names: Sequence[str], reset: bool) -> None:
    with conn.cursor() as cur:
        if reset:
            for table_name in table_names:
                cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
            conn.commit()
            return

        for table_name in table_names:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            if count > 0:
                raise SystemExit(
                    f"Table {table_name} already has {count:,} rows. "
                    "Use --reset if you want to rebuild it."
                )


def insert_josaa(conn, batch_size: int = 5000) -> None:
    insert_sql = """
        INSERT INTO josaa_btech_2024 (
            year, round, institute, institute_type, program, quota,
            category, gender, opening_rank, closing_rank
        ) VALUES %s
    """
    batch: List[Tuple[int, int, str, str, str, str, str, str, str, str]] = []
    total = 0
    current_round = None
    with conn.cursor() as cur:
        for round_num, row in iter_josaa_rows():
            if current_round is None:
                current_round = round_num
            elif round_num != current_round and batch:
                execute_values(cur, insert_sql, batch, page_size=min(batch_size, len(batch)))
                conn.commit()
                total += len(batch)
                print(f"Inserted JoSAA rows: {total:,}")
                batch.clear()
                current_round = round_num
            batch.append(row)
            if len(batch) >= batch_size:
                execute_values(cur, insert_sql, batch, page_size=batch_size)
                conn.commit()
                total += len(batch)
                print(f"Inserted JoSAA rows: {total:,}")
                batch.clear()
        if batch:
            execute_values(cur, insert_sql, batch, page_size=len(batch))
            conn.commit()
            total += len(batch)
            print(f"Inserted JoSAA rows: {total:,}")


def insert_nirf(conn, csv_path: Path, table_name: str) -> None:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            rows.append((
                int(record["year"]),
                record.get("category", ""),
                int(record["rank"]),
                normalize_spaces(record.get("institute_name_raw", "")),
                normalize_spaces(record.get("institute_name_norm", "")),
                record.get("institute_id", ""),
                record.get("state", ""),
                record.get("city", ""),
                float(record["score"]) if record.get("score") else None,
                record.get("page_url", ""),
                record.get("row_anchor", ""),
                record.get("crawl_time", ""),
            ))

    sql = f"""
        INSERT INTO {table_name} (
            year, category, rank, institute, institute_normalized, institute_id,
            state, city, score, page_url, row_anchor, crawl_time
        ) VALUES %s
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    print(f"Inserted NIRF rows: {len(rows):,}")


def load_alias_csv(alias_csv: Path) -> List[Tuple[str, str, float]]:
    if not alias_csv.exists():
        return []
    rows = []
    with alias_csv.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            alias = normalize_spaces(record.get("alias", ""))
            canonical = normalize_spaces(record.get("canonical_norm", ""))
            if not alias or not canonical:
                continue
            rows.append((alias, canonical, 0.9))
    return rows


def rebuild_mappings(conn, alias_csv: Path, nirf_table: str) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE institute_mapping RESTART IDENTITY")

        cur.executemany(
            """
            INSERT INTO institute_mapping (josaa_name, nirf_name, confidence_score)
            VALUES (%s, %s, %s)
            ON CONFLICT (josaa_name) DO NOTHING
            """,
            MANUAL_MAPPINGS,
        )

        alias_rows = load_alias_csv(alias_csv)
        if alias_rows:
            cur.executemany(
                """
                INSERT INTO institute_mapping (josaa_name, nirf_name, confidence_score)
                VALUES (%s, %s, %s)
                ON CONFLICT (josaa_name) DO NOTHING
                """,
                alias_rows,
            )

        cur.execute(
            f"""
            INSERT INTO institute_mapping (josaa_name, nirf_name, confidence_score)
            SELECT DISTINCT
                j.institute AS josaa_name,
                n.institute AS nirf_name,
                0.9 AS confidence_score
            FROM (SELECT DISTINCT institute FROM josaa_btech_2024) j
            CROSS JOIN (SELECT DISTINCT institute FROM {nirf_table}) n
            WHERE regexp_replace(lower(trim(j.institute)), '\s+', ' ', 'g')
                  = regexp_replace(lower(trim(n.institute)), '\s+', ' ', 'g')
              AND NOT EXISTS (
                  SELECT 1 FROM institute_mapping im WHERE im.josaa_name = j.institute
              )
            """
        )
    conn.commit()


def print_counts(conn, nirf_table: str) -> None:
    with conn.cursor() as cur:
        for table_name in ("josaa_btech_2024", nirf_table, "institute_mapping"):
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            print(f"{table_name}: {cur.fetchone()[0]:,} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Supabase schema and data.")
    parser.add_argument("--nirf-csv", type=Path, default=DEFAULT_NIRF_CSV)
    parser.add_argument("--alias-csv", type=Path, default=DEFAULT_ALIAS_CSV)
    parser.add_argument("--skip-josaa", action="store_true")
    parser.add_argument("--skip-nirf", action="store_true")
    parser.add_argument("--skip-mapping", action="store_true")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate target tables before loading. Required if rerunning after data already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nirf_csv = args.nirf_csv.resolve()
    if not args.skip_nirf and not nirf_csv.exists():
        raise SystemExit(f"NIRF CSV not found: {nirf_csv}")

    nirf_table = nirf_table_name(nirf_csv) if nirf_csv.exists() else "nirf_rankings_2025"
    params = db_params()
    print(f"Connecting to {params['host']}:{params['port']} / {params['database']}")
    conn = psycopg2.connect(**params)
    try:
        ensure_schema(conn, nirf_table)
        ensure_empty_or_reset(
            conn,
            ["josaa_btech_2024", nirf_table, "institute_mapping"],
            reset=args.reset,
        )

        if not args.skip_josaa:
            insert_josaa(conn)
        if not args.skip_nirf:
            insert_nirf(conn, nirf_csv, nirf_table)
        if not args.skip_mapping:
            rebuild_mappings(conn, args.alias_csv.resolve(), nirf_table)

        ensure_schema(conn, nirf_table)
        print_counts(conn, nirf_table)
        print("Bootstrap completed successfully.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
