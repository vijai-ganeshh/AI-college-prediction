import streamlit as st
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import google.generativeai as genai
import logging
import traceback
from io import StringIO
import sys
import os
import json
import html
from pathlib import Path
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========================
# LOGGING SETUP
# ========================
def setup_logging():
    logger = logging.getLogger('josaa_rag')
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    string_handler = logging.StreamHandler(st.session_state.get('log_stream', StringIO()))
    string_handler.setLevel(logging.DEBUG)
    string_handler.setFormatter(fmt)
    logger.addHandler(string_handler)
    return logger

if 'log_stream' not in st.session_state:
    st.session_state.log_stream = StringIO()

logger = setup_logging()

def log_performance(name, start):
    dur = time.time() - start
    logger.info(f"⏱️  {name} completed in {dur:.2f}s")
    return dur

def log_error(op, e):
    logger.error(f"❌ {op} failed: {e}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")

# ========================
# CREDENTIALS (ENV-FIRST)
# ========================
def load_local_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        log_error("Local env load", e)


load_local_env()
ROOT_DIR = Path(__file__).resolve().parent

DB = {
    "host": os.getenv("DB_HOST", "DB_HOST_HERE"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "DB_USER_HERE"),
    "password": os.getenv("DB_PASSWORD", "DB_PASSWORD_HERE"),
    "port": os.getenv("DB_PORT", "6543"),
    "sslmode": os.getenv("DB_SSLMODE", "require"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
}
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY_HERE")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "coderop12/gemma2b-nirf-lookup-2025")
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "")
USE_HF_SUMMARY = os.getenv("USE_HF_SUMMARY", "true").strip().lower() in {"1", "true", "yes", "on"}

logger.info("🔧 Configuration loaded")

# ========================
# POLICY
# ========================
@dataclass
class Policy:
    exclude_pwd_default: bool = True
    final_round_default: bool = True
    apply_open_gn_ai_on_air: bool = True
    apply_open_gn_ai_on_numeric_eligibility: bool = True

# ========================
# CONSTANTS & HELPERS
# ========================
RANK_NUM_EXPR = (
    "CAST(NULLIF(regexp_replace(trim(closing_rank), '[^0-9]', '', 'g'), '') AS INTEGER)"
)
ALLOWED_READ = re.compile(r"^\s*(select|with)\b", re.IGNORECASE | re.DOTALL)
DANGEROUS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|vacuum|analyze)\b",
    re.IGNORECASE,
)
EAMCET_PATTERN = re.compile(r"\b(?:ap|ts|tg)?\s*ea(?:m|p)cet\b", re.IGNORECASE)
AP_EAMCET_PATTERN = re.compile(r"\bap\s*ea(?:m|p)cet\b", re.IGNORECASE)
TS_EAMCET_PATTERN = re.compile(r"\b(?:ts|tg)\s*ea(?:m|p)cet\b", re.IGNORECASE)

def single_statement(sql: str) -> bool:
    return sql.strip().count(";") <= 1

def ensure_limit(sql: str, hard_limit: int = 15) -> str:
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip(';')} LIMIT {hard_limit};"

def sanitize_select(sql: str, hard_limit: int = 15) -> str:
    sql = sql.strip()
    if not single_statement(sql):
        raise ValueError("Multiple SQL statements not allowed.")
    if not ALLOWED_READ.match(sql):
        raise ValueError("Only SELECT/CTE read queries are allowed.")
    if DANGEROUS.search(sql):
        raise ValueError("Potentially dangerous SQL token detected.")
    return ensure_limit(sql, hard_limit)

def extract_rank_value(text: str) -> Optional[int]:
    patterns = [
        r'\b(?:jee\s*(?:main|mains|advanced)\s*)?(?:air|rank)\s*(?:of|is|=|:)?\s*(\d{1,6})\b',
        r'\b(\d{3,6})\b',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return int(m.group(1))
    return None

def detect_exam_type(text: str) -> str:
    q = text.lower()
    if "jee advanced" in q or "advanced rank" in q:
        return "jee_advanced"
    if "jee main" in q or "jee mains" in q or "mains rank" in q:
        return "jee_main"
    return ""

def detect_web_only_exam(text: str) -> str:
    if AP_EAMCET_PATTERN.search(text):
        return "ap_eamcet"
    if TS_EAMCET_PATTERN.search(text):
        return "ts_eamcet"
    if EAMCET_PATTERN.search(text):
        return "eamcet"
    return ""

def web_only_exam_label(exam_kind: str) -> str:
    labels = {
        "ap_eamcet": "AP EAMCET / AP EAPCET",
        "ts_eamcet": "TS / TG EAMCET / EAPCET",
        "eamcet": "EAMCET / EAPCET",
    }
    return labels.get(exam_kind, "web-only counselling")

def detect_year_value(text: str) -> int:
    m = re.search(r'\b(20\d{2})\b', text)
    if m:
        return int(m.group(1))
    return 2024

def detect_round_value(text: str) -> Optional[int]:
    m = re.search(r'\bround\s*(\d+)\b', text, re.I)
    if m:
        return int(m.group(1))
    return None

def detect_quota(text: str) -> str:
    q = text.lower()
    if re.search(r'\bhs\b|\bhome state\b', q):
        return "HS"
    if re.search(r'\bos\b|\bother state\b', q):
        return "OS"
    if re.search(r'\bai\b|\ball india\b', q):
        return "AI"
    return "AI"

def detect_gender(text: str) -> str:
    q = text.lower()
    if "female-only" in q or "female only" in q or "girls" in q:
        return "Female-only (including Supernumerary)"
    return "Gender-Neutral"

def detect_category(text: str) -> str:
    q = text.lower()
    if "obc" in q:
        return "OBC-NCL"
    if "ews" in q:
        return "GEN-EWS"
    if re.search(r'\bsc\b', q):
        return "SC"
    if re.search(r'\bst\b', q):
        return "ST"
    return "OPEN"

def detect_institute_types(text: str, exam_type: str) -> List[str]:
    q = text.lower()
    picked = []
    mapping = {
        "IIT": ["iit", "indian institute of technology"],
        "NIT": ["nit", "national institute of technology"],
        "IIIT": ["iiit", "indian institute of information technology"],
        "GFTI": ["gfti"],
    }
    for inst_type, tokens in mapping.items():
        if any(token in q for token in tokens):
            picked.append(inst_type)
    if picked:
        if exam_type == "jee_main":
            return [t for t in picked if t != "IIT"] or ["NIT", "IIIT", "GFTI"]
        return picked
    if exam_type == "jee_main":
        return ["NIT", "IIIT", "GFTI"]
    return []

def extract_program_patterns(text: str) -> List[str]:
    q = text.lower()
    families = [
        (["computer science", "cse"], [
            "%Computer Science%",
            "%Information Technology%",
            "%Artificial Intelligence%",
            "%Data Science%",
            "%Software Engineering%",
        ]),
        (["electronics and communication", "ece"], [
            "%Electronics and Communication%",
            "%Electronics and Telecommunication%",
        ]),
        (["electrical", "eee"], ["%Electrical%"]),
        (["mechanical"], ["%Mechanical%"]),
        (["civil"], ["%Civil%"]),
        (["chemical"], ["%Chemical%"]),
        (["ai", "artificial intelligence"], ["%Artificial Intelligence%", "%AI and Data Science%"]),
    ]
    for tokens, patterns in families:
        if any(token in q for token in tokens):
            return patterns
    return []

def recommendation_intent(text: str) -> bool:
    rank = extract_rank_value(text)
    if not rank:
        return False
    return bool(re.search(
        r"\b(suggest|recommend|options|colleges?|can i get|what can i get|which colleges?|which college)\b",
        text,
        re.I,
    ))

def deterministic_exam_warning(text: str) -> Optional[str]:
    q = text.lower()
    if detect_exam_type(text) == "jee_main" and ("iit" in q or "indian institute of technology" in q):
        return (
            "IIT admissions are based on JEE Advanced ranks, not JEE Main rank alone. "
            "Ask with your JEE Advanced rank for IIT options, or ask for NIT/IIIT/GFTI options using JEE Main rank."
        )
    return None

def build_deterministic_sql(user_query: str) -> Optional[str]:
    if not recommendation_intent(user_query):
        return None

    rank = extract_rank_value(user_query)
    if not rank:
        return None

    year = detect_year_value(user_query)
    round_value = detect_round_value(user_query)
    exam_type = detect_exam_type(user_query)
    institute_types = detect_institute_types(user_query, exam_type)
    program_patterns = extract_program_patterns(user_query)
    category = detect_category(user_query)
    quota = detect_quota(user_query)
    gender = detect_gender(user_query)
    lower_rank = max(1, rank - 4000)
    upper_rank = rank + 20000

    where = [
        f"year = {year}",
        "closing_rank_num IS NOT NULL",
        f"category = '{category}'",
        f"quota = '{quota}'",
        f"gender = '{gender}'",
        "category NOT ILIKE '%PwD%'",
    ]
    if round_value is None:
        where.append(f"round = (SELECT MAX(round) FROM josaa_btech_2024 WHERE year = {year})")
    else:
        where.append(f"round = {round_value}")
    if institute_types:
        where.append("institute_type IN (" + ", ".join(f"'{x}'" for x in institute_types) + ")")
    if program_patterns:
        where.append("(" + " OR ".join(f"program ILIKE '{p}'" for p in program_patterns) + ")")
    where.append(f"closing_rank_num BETWEEN {lower_rank} AND {upper_rank}")

    return f"""
WITH ranked AS (
    SELECT
        year,
        round,
        institute,
        institute_type,
        program,
        quota,
        category,
        gender,
        opening_rank,
        closing_rank,
        {RANK_NUM_EXPR} AS closing_rank_num
    FROM josaa_btech_2024
)
SELECT
    year,
    round,
    institute,
    institute_type,
    program,
    quota,
    category,
    gender,
    opening_rank,
    closing_rank,
    closing_rank_num
FROM ranked
WHERE {" AND ".join(where)}
ORDER BY
    CASE WHEN closing_rank_num >= {rank} THEN 0 ELSE 1 END,
    ABS(closing_rank_num - {rank}),
    closing_rank_num ASC NULLS LAST
LIMIT 15;
""".strip()

def air_context_hint(text: str) -> str:
    return ("User intent hint: Eligibility by AIR is requested; apply OPEN/GN/AI, "
            "exclude PwD, and latest round unless the user overrides."
            ) if re.search(r'\b(AIR|rank)\b', text, re.I) else ""

def numeric_eligibility_hint(text: str) -> bool:
    return bool(
        re.search(r'\b(under|below|less\s+than|greater\s+than|over|at\s+least|at\s+most)\s*\d+', text, re.I)
        or re.search(r'\b(<=|>=|<|>)\s*\d+', text)
        or re.search(r'\bclosing\s*rank\b.*\d+', text, re.I)
    )

def fix_distinct_orderby(sql: str) -> str:
    if re.search(r"\bselect\s+distinct\b", sql, re.IGNORECASE) and \
       re.search(r"\border\s+by\s+.*closing_rank_num", sql, re.IGNORECASE):
        m = re.search(r"(?is)select\s+distinct\s+(.*?)\s+from\s", sql)
        if m and "closing_rank_num" not in m.group(1):
            start, end = m.span(1)
            sql = sql[:start] + m.group(1).rstrip() + ", closing_rank_num" + sql[end:]
    return sql

def _rank_to_num(v: str) -> int:
    s = re.sub(r'[^0-9]', '', str(v or ''))
    return int(s) if s else 0

def _round_to_int(v) -> int:
    try:
        return int(v)
    except Exception:
        return -1

def _norm(x): return (x or '').strip().lower()

def _key_tuple(r: Dict) -> tuple:
    return (_norm(r.get("institute")), _norm(r.get("program")),
            _norm(r.get("category")), _norm(r.get("quota")), _norm(r.get("gender")))

def _sort_tuple(r: Dict) -> tuple:
    return (_rank_to_num(r.get("closing_rank")),
            -_round_to_int(r.get("round")),
            -(int(r.get("year")) if str(r.get("year") or '').isdigit() else -1))


def wants_web_enrichment(text: str) -> bool:
    return bool(re.search(
        r"\b(suggest|recommend|best|compare|worth|review|placement|placements|fees|fee|hostel|campus|curriculum|college|colleges)\b",
        text,
        re.I,
    ))

def rank_note_for_query(text: str) -> Optional[str]:
    if detect_exam_type(text) == "jee_main":
        return "JEE Main rank usually maps to NIT/IIIT/GFTI counselling. IIT admissions typically use JEE Advanced rank."
    return None

def clean_generated_text(text: str) -> str:
    if not text:
        return text
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("\\_", "_")
    cleaned = re.sub(r'closing[_ ]rank\s*=\s*', 'closing rank ', cleaned, flags=re.I)
    cleaned = re.sub(r'\|\s*closing rank\s*', ' (closing rank ', cleaned, flags=re.I)
    cleaned = re.sub(r'\)\s*\)', ')', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

# ========================
# DB LAYER
# ========================
class Pg:
    def __init__(self, params: Dict[str, str]):
        logger.info("🗃️  Initializing database connection pool")
        start = time.time()
        try:
            self.pool = SimpleConnectionPool(minconn=1, maxconn=6, **params)
            log_performance("Database pool creation", start)
        except Exception as e:
            log_error("Database pool creation", e); raise

    def _conn(self):
        start = time.time()
        try:
            conn = self.pool.getconn()
            with conn.cursor() as c:
                c.execute("SET LOCAL statement_timeout = '12000ms';")
                c.execute("SET LOCAL default_transaction_read_only = on;")
            log_performance("Database connection setup", start)
            return conn
        except Exception as e:
            log_error("Database connection", e); raise

    def fetch_schema_text(self) -> str:
        logger.info("📊 Fetching live database schema")
        start = time.time()
        conn = self._conn()
        try:
            cur = conn.cursor()
            out = ["SCHEMA: public\n"]
            cur.execute("""SELECT table_name FROM information_schema.tables
                           WHERE table_schema='public' ORDER BY table_name;""")
            tables = cur.fetchall()
            for (t,) in tables:
                out.append(f"TABLE: {t}")
                cur.execute("""SELECT column_name, data_type, is_nullable
                               FROM information_schema.columns
                               WHERE table_schema='public' AND table_name=%s
                               ORDER BY ordinal_position;""", (t,))
                for name, dtype, nullable in cur.fetchall():
                    out.append(f"  - {name}: {dtype} {'NULL' if nullable=='YES' else 'NOT NULL'}")
                out.append("")
            schema = "\n".join(out)
            log_performance("Schema fetch", start)
            logger.info(f"✅ Schema fetched successfully ({len(schema)} chars)")
            return schema
        except Exception as e:
            log_error("Schema fetch", e); raise
        finally:
            self.pool.putconn(conn)

    def run(self, sql: str) -> List[Dict]:
        logger.info("🏃‍♀️ Executing SQL query")
        start = time.time()
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
            log_performance("SQL execution", start)
            logger.info(f"✅ Query executed: {len(rows)} rows")
            return rows
        except Exception as e:
            log_error("SQL execution", e); raise
        finally:
            self.pool.putconn(conn)


class TavilySearch:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self.enabled = bool(self.api_key)
        if self.enabled:
            logger.info("🌐 Tavily web search enabled")
        else:
            logger.info("🌐 Tavily web search disabled (no API key)")

    def enrich(self, user_query: str, rows: List[Dict], max_queries: int = 3) -> List[Dict]:
        if not self.enabled or not rows or not wants_web_enrichment(user_query):
            return []

        searches = []
        seen = set()
        for row in rows:
            inst = (row.get("institute") or "").strip()
            prog = (row.get("program") or "").strip()
            if not inst:
                continue
            key = (inst.lower(), prog.lower())
            if key in seen:
                continue
            seen.add(key)
            searches.append({
                "query": self._build_query(user_query, inst, prog),
                "institute": inst,
                "program": prog,
            })
            if len(searches) >= max_queries:
                break

        results = []
        for item in searches:
            for hit in self._search(item["query"], max_results=3):
                hit["institute"] = item["institute"]
                hit["program"] = item["program"]
                results.append(hit)
        return self._dedup_results(results)

    def search_general(self, user_query: str, max_results: int = 6) -> List[Dict]:
        if not self.enabled or not user_query.strip():
            return []
        results = self._search(user_query.strip(), max_results=max_results)
        return self._dedup_results(results)

    def search_exam_guidance(self, user_query: str, exam_kind: str, max_queries: int = 3) -> List[Dict]:
        if not self.enabled or not user_query.strip():
            return []

        queries = []
        seen = set()
        for query in self._build_exam_queries(user_query, exam_kind)[:max_queries]:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            queries.append(query)

        results = []
        for idx, query in enumerate(queries):
            per_query = 4 if idx == 0 else 3
            results.extend(self._search(query, max_results=per_query))
        return self._dedup_results(results)

    def _build_query(self, user_query: str, institute: str, program: str) -> str:
        parts = [institute]
        if program:
            parts.append(program)
        parts.extend(["official website", "placements", "fees", "curriculum"])
        return " ".join(parts)

    def _build_exam_queries(self, user_query: str, exam_kind: str) -> List[str]:
        label = web_only_exam_label(exam_kind)
        rank = extract_rank_value(user_query)
        queries = [
            user_query.strip(),
            f"{label} counselling official website cutoffs colleges {user_query.strip()}",
        ]
        if recommendation_intent(user_query):
            if rank:
                queries.append(f"{label} {rank} rank colleges cutoff predictor counselling")
            else:
                queries.append(f"{label} best colleges counselling cutoff predictor")
        else:
            queries.append(f"{label} official counselling cutoff rank list {user_query.strip()}")
        return queries

    def _domain_rank(self, domain: str) -> int:
        domain = (domain or "").lower()
        if any(token in domain for token in ("youtube.com", "youtu.be", "quora.com", "reddit.com")):
            return -1
        if any(token in domain for token in ("aptonline.in", "tgche.ac.in")):
            return 4
        if any(domain.endswith(suffix) for suffix in (".gov.in", ".nic.in")):
            return 4
        if any(domain.endswith(suffix) for suffix in (".ac.in", ".edu", ".edu.in")):
            return 3
        if any(token in domain for token in ("josaa.nic.in", "nirfindia.org", "iit", "nit", "iiit")):
            return 2
        if any(token in domain for token in ("shiksha", "collegedunia", "careers360", "campusoption", "collegedekho")):
            return 0
        return 1

    def _search(self, query: str, max_results: int = 3) -> List[Dict]:
        payload = json.dumps({
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }).encode("utf-8")
        req = urlrequest.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hits = data.get("results", [])
            return [{
                "title": h.get("title", ""),
                "url": h.get("url", ""),
                "content": h.get("content", ""),
                "score": h.get("score", 0),
                "domain": urlparse.urlparse(h.get("url", "")).netloc,
            } for h in hits]
        except urlerror.URLError as e:
            log_error("Tavily search", e)
            return []
        except Exception as e:
            log_error("Tavily search parse", e)
            return []

    def _dedup_results(self, results: List[Dict]) -> List[Dict]:
        out, seen = [], set()
        ranked = sorted(
            results,
            key=lambda item: (
                -self._domain_rank(item.get("domain", "")),
                -float(item.get("score", 0) or 0),
            ),
        )
        low_signal_count = 0
        for item in ranked:
            url = (item.get("url") or "").strip().lower()
            if not url or url in seen:
                continue
            if self._domain_rank(item.get("domain", "")) < 0:
                continue
            if self._domain_rank(item.get("domain", "")) == 0:
                low_signal_count += 1
                if low_signal_count > 2:
                    continue
            seen.add(url)
            out.append(item)
        return out[:8]


class HFSummaryDraftModel:
    def __init__(self, enabled: bool, model_id: str, token: str, local_path: str = ""):
        self.enabled = enabled and bool(model_id)
        self.model_id = (model_id or "").strip()
        self.token = (token or "").strip()
        self.local_path = (local_path or "").strip()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.failed = False
        if self.enabled:
            logger.info("🤗 HF summary drafting enabled")
        else:
            logger.info("🤗 HF summary drafting disabled")

    def draft(self, user_query: str, rows: List[Dict], fallback_text: str) -> str:
        if not self.enabled or self.failed or not rows:
            return fallback_text
        try:
            self._ensure_loaded()
            prompt = self._build_prompt(user_query, rows, fallback_text)
            rendered = self._render_prompt(prompt)
            inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1800)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=320,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return text or fallback_text
        except Exception as e:
            self.failed = True
            log_error("HF summary drafting", e)
            return fallback_text

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        source = self._resolve_source()
        logger.info(f"🤗 Loading HF summary model from {source}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(source, token=self.token or None)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            token=self.token or None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_source(self) -> str:
        candidates = []
        if self.local_path:
            candidates.append(Path(self.local_path))
        candidates.extend([
            ROOT_DIR / "gemma2b-nirf-lookup",
            ROOT_DIR / "gemma2b-nirf-model",
            ROOT_DIR / "hf_model",
        ])
        for candidate in candidates:
            if candidate.exists() and (candidate / "config.json").exists():
                return str(candidate)
        return snapshot_download(repo_id=self.model_id, token=self.token or None)

    def _build_prompt(self, user_query: str, rows: List[Dict], fallback_text: str) -> str:
        row_lines = []
        for row in rows[:12]:
            closing_rank = row.get("closing_rank") or row.get("closing_rank_num") or ""
            row_lines.append(
                f"- Institute: {row.get('institute','')} | Type: {row.get('institute_type','')} | "
                f"Program: {row.get('program','')} | Quota: {row.get('quota','')} | "
                f"Category: {row.get('category','')} | Gender: {row.get('gender','')} | "
                f"Round: {row.get('round','')} | Closing rank: {closing_rank}"
            )
        return f"""
You are drafting a concise JoSAA counselling summary.

Rules:
- Use only the structured rows.
- Do not use web information.
- Do not invent colleges or ranks.
- Prefer concise markdown.
- Mention feasibility relative to the user's rank when possible.
- Do not emit raw keys like closing_rank= or HTML entities.

User query:
{user_query}

Structured rows:
{chr(10).join(row_lines)}

Fallback draft:
{fallback_text}
""".strip()

    def _render_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

# ========================
# STAGE A: QUERY ENHANCER
# ========================
class QueryEnhancer:
    def __init__(self, api_key: str, policy: Policy):
        logger.info("🔍 Initializing Query Enhancer (Gemini Flash)")
        start = time.time()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.policy = policy
        log_performance("Query Enhancer initialization", start)

    def enhance(self, user_query: str) -> List[str]:
        logger.info(f"🔍 Enhancing user query: '{user_query}'")
        start = time.time()
        if recommendation_intent(user_query):
            return [user_query]
        hints = []
        if self.policy.exclude_pwd_default:
            hints.append("exclude PwD by default unless the user explicitly asks for PwD")
        if self.policy.final_round_default:
            hints.append("assume final (latest) JoSAA round when round is unspecified")
        if self.policy.apply_open_gn_ai_on_numeric_eligibility:
            hints.append("for numeric cutoff queries assume OPEN, Gender-Neutral, quota AI unless specified")
        generic = "; ".join(hints) if hints else "no additional defaults"
        air_hint = ("If the user mentions AIR but not category/gender/quota/round, append "
                    "(assume OPEN, Gender-Neutral, quota AI, exclude PwD, final round)."
                   ) if self.policy.apply_open_gn_ai_on_air else ""
        prompt = f"""
Rewrite the user's database question into 2–3 concise variants.
- Keep the same intent, one line each.
- Do NOT invent constraints.
- When unspecified, {generic}.
- {air_hint}

User query: "{user_query}"
Variants:
"""
        try:
            res = self.model.generate_content(prompt)
            text = (res.text or "").strip()
            variants = self._extract_variants(text, user_query)
            variants = variants[:3] if variants else [user_query]
            log_performance("Query enhancement", start)
            return variants
        except Exception as e:
            log_error("Query enhancement", e)
            return [user_query]

    def _extract_variants(self, text: str, user_query: str) -> List[str]:
        variants = []
        for raw in text.splitlines():
            line = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', raw).strip().strip('"')
            if not line:
                continue
            if re.match(r'(?i)^(variants?|here are|rewrite|user query|query:)', line):
                continue
            if len(line) < 8:
                continue
            variants.append(line)
        deduped = []
        seen = set()
        for line in variants:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)
        return deduped or [user_query]

# ========================
# STAGE B: SQL GEN & CRITIQUE
# ========================
class SqlGenPro:
    def __init__(self, api_key: str, schema_text: str, policy: Policy):
        logger.info("🧠 Initializing SQL Generator (Gemini 2.5 Flash)")
        start = time.time()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.schema_text = schema_text
        self.policy = policy
        log_performance("SQL Generator initialization", start)

    def _rules(self) -> str:
        base = f"""
DATA RULES:
- opening_rank and closing_rank are TEXT. Use:
  {RANK_NUM_EXPR} AS closing_rank_num
- Use ILIKE for case-insensitive matches.
- Prefer table 'josaa_btech_2024' for B.Tech queries.
- IIT filter: institute ILIKE '%Indian Institute of Technology%' OR institute ILIKE 'IIT %'
- Include year and round in SELECT when present.
- If you use closing_rank_num in ORDER BY/filters, include it in SELECT.
- ORDER BY closing_rank_num ASC NULLS LAST.
- ALWAYS LIMIT 15.
"""
        extra = []
        if self.policy.exclude_pwd_default:
            extra.append("If user didn't mention PwD, add: category NOT ILIKE '%PwD%'.")
        if self.policy.final_round_default:
            extra.append("If round unspecified, add: round = (SELECT MAX(round) FROM josaa_btech_2024 WHERE year=2024).")
        if self.policy.apply_open_gn_ai_on_air:
            extra.append("If asking by AIR and unspecified, add quota='AI', category='OPEN', category NOT ILIKE '%PwD%', gender='Gender-Neutral', round=max for 2024; filter closing_rank_num >= AIR.")
        if self.policy.apply_open_gn_ai_on_numeric_eligibility:
            extra.append("If numeric cutoff but no category/gender/quota, add quota='AI', category='OPEN', category NOT ILIKE '%PwD%', gender='Gender-Neutral'.")
        extra.append("Never rewrite JEE Main rank queries into IIT eligibility unless the user explicitly asks for JEE Advanced.")
        extra.append("For recommendation queries with a numeric rank, keep the rank value unchanged and avoid inventing narrower bands like 5000-6000.")
        if extra:
            base += "- " + "\n- ".join(extra) + "\n"
        return base

    def to_sql(self, nl_query: str) -> Optional[str]:
        logger.info(f"🛠️  Generating SQL for: '{nl_query}'")
        start = time.time()
        deterministic_sql = build_deterministic_sql(nl_query)
        if deterministic_sql:
            logger.info("🧭 Using deterministic SQL path for rank recommendation query")
            return deterministic_sql
        rules = self._rules()
        ctx = []
        if air_context_hint(nl_query):
            ctx.append(air_context_hint(nl_query))
        if self.policy.apply_open_gn_ai_on_numeric_eligibility and numeric_eligibility_hint(nl_query):
            ctx.append("Numeric cutoff intent detected; apply OPEN/GN/AI, exclude PwD, latest round.")
        ctx = "\n".join(ctx)

        prompt = f"""
You are an expert PostgreSQL writer.

DATABASE SCHEMA (public):
{self.schema_text}

{rules}

{ctx}

Return ONLY a single SQL SELECT (WITH allowed).

User query: "{nl_query}"
SQL:
"""
        try:
            res = self.model.generate_content(prompt)
            sql = (res.text or "").strip()
            if sql.startswith("```"):
                sql = sql.replace("```sql", "").replace("```", "").strip()
            if not sql:
                return None
            sql = fix_distinct_orderby(sql)
            # Critique/fix
            fix = self.critique_fix(sql, nl_query, rules)
            sql_out = fix or sql
            log_performance("SQL generation", start)
            return sql_out
        except Exception as e:
            log_error("SQL generation", e)
            return None

    def critique_fix(self, sql: str, nl_query: str, rules: str) -> Optional[str]:
        start = time.time()
        critic = f"""
You are a PostgreSQL critic. Ensure:
- closing_rank_num derived & in SELECT if used,
- exclude PwD by default,
- final round if unspecified,
- OPEN/GN/AI defaults when AIR/eligibility unspecified,
- IIT name via ILIKE,
- include year & round,
- ORDER BY closing_rank_num ASC NULLS LAST,
- LIMIT 15,
- single SELECT/WITH.
If OK, return original; else corrected.

User: "{nl_query}"
SQL:
{sql}

Return ONLY SQL:
"""
        try:
            res = self.model.generate_content(critic)
            fix = (res.text or "").strip()
            if fix and fix != sql:
                if fix.startswith("```"):
                    fix = fix.replace("```sql", "").replace("```", "").strip()
                log_performance("SQL critique", start)
                return fix
            log_performance("SQL critique", start)
            return None
        except Exception as e:
            log_error("SQL critique", e)
            return None

# ========================
# STAGE C: CHAT ANSWER (NO TABLES)
# ========================
class Answerer:
    def __init__(self, api_key: str, summary_drafter: Optional[HFSummaryDraftModel] = None):
        logger.info("💬 Initializing Answer Generator")
        start = time.time()
        genai.configure(api_key=api_key)
        self.sim = genai.GenerativeModel("gemini-2.5-flash")
        self.summary_drafter = summary_drafter
        log_performance("Answer Generator initialization", start)

    def answer(self, user_query: str, rows: List[Dict], web_results: Optional[List[Dict]] = None) -> str:
        logger.info(f"💬 Generating answer for {len(rows)} rows")
        start = time.time()
        if not rows:
            return ("### 🔍 No matches yet\n\n"
                    "Try widening the rank window (±1000), consider more categories, "
                    "check earlier rounds, and add NIT/IIIT backups.")
        data = self._analyze(rows)
        base = self._compose(user_query, data, rows)
        if self.summary_drafter:
            base = self.summary_drafter.draft(user_query, rows, base)
        base = clean_generated_text(base)
        txt = self._refine_final_answer(user_query, base, rows, web_results or [])
        txt = clean_generated_text(txt)
        log_performance("Answer generation", start)
        return txt

    def answer_from_web_only(self, user_query: str, web_results: List[Dict], exam_kind: str = "") -> str:
        exam_label = web_only_exam_label(exam_kind) if exam_kind else ""
        if not web_results:
            if exam_label:
                return (
                    f"### {exam_label} web lookup\n\n"
                    "This query was kept out of the JoSAA database because it belongs to a different counselling system. "
                    "I could not get enough reliable Tavily web results to answer it confidently yet. "
                    "Try adding your state, branch, category, or counselling year."
                )
            return (
                "### No reliable result yet\n\n"
                "I could not get enough structured data or useful web results for that query. "
                "Try adding the institute name, branch, exam, rank, or category."
            )

        web_lines = []
        for item in web_results[:8]:
            web_lines.append(
                f"- Title: {item.get('title', '')} | URL: {item.get('url', '')} | "
                f"{item.get('content', '')[:320]}"
            )

        exam_rules = ""
        if exam_label:
            exam_rules = f"""
- The user is asking about {exam_label}.
- The local database contains only JoSAA cutoff data, so it must not be used or referenced as evidence here.
- Do not reinterpret {exam_label} ranks as JEE Main, JEE Advanced, JoSAA, or IIT ranks.
- If the web results suggest colleges/cutoffs, present them as likely or indicative unless the source clearly gives official cutoff data.
- Mention that final options can vary by branch, category, gender, locality quota, and counselling phase/round when relevant.
"""

        prompt = f"""
        You are helping with college counselling using web search results only.

        Rules:
        - Answer using only the supplied web search results.
        - Do not claim structured JoSAA cutoff certainty when no database evidence is provided.
        - If the query asks about fees, placements, campus, comparison, reviews, or broad suggestions, synthesize from the sources.
        - If the query is about admissions and you do not have cutoff evidence, state that clearly.
        - Prefer concise markdown.
        - Cite sources inline with URLs when using web facts.
        {exam_rules}

        User query:
        {user_query}

Web search results:
{chr(10).join(web_lines)}
"""
        try:
            res = self.sim.generate_content(prompt)
            text = (res.text or "").strip()
            return clean_generated_text(text) or self._append_web_sources("", web_results)
        except Exception as e:
            log_error("Web-only answer generation", e)
            return self._append_web_sources(
                "### Web fallback\n\nI could not complete the structured pipeline for this query, so here are the most relevant web sources I found.",
                web_results,
            )

    def _analyze(self, rows: List[Dict]) -> Dict:
        inst = {}
        progs = set()
        rmin, rmax = float('inf'), 0
        for r in rows:
            i = r.get('institute', 'Unknown')
            p = r.get('program', 'Unknown')
            cr_val = r.get('closing_rank')
            if cr_val in (None, "") and r.get('closing_rank_num') not in (None, ""):
                cr_val = r.get('closing_rank_num')
            crs = str(cr_val or '0')
            cr = int(''.join(filter(str.isdigit, crs)) or '0')
            if cr > 0:
                rmin = min(rmin, cr); rmax = max(rmax, cr)
            progs.add(p)
            inst.setdefault(i, []).append({
                'program': p, 'closing_rank': crs, 'rank_num': cr,
                'category': r.get('category',''), 'quota': r.get('quota',''),
                'gender': r.get('gender',''), 'round': r.get('round',''),
                'year': r.get('year','')
            })
        for k in inst: inst[k].sort(key=lambda x: x['rank_num'] or 10**9)
        return {'institutes': inst, 'programs': list(progs),
                'rank_range': (rmin if rmin != float('inf') else 0, rmax),
                'total': len(rows)}

    def _extract_rank(self, q: str) -> Optional[int]:
        return extract_rank_value(q)

    def _bucket(self, ur: Optional[int], cr: int) -> str:
        if not ur or not cr: return ""
        diff = cr - ur
        if diff >= 1500: return "Likely"
        if diff > -1500: return "Balanced"
        return "Reach"

    def _compose(self, user_query: str, data: Dict, raw: List[Dict]) -> str:
        ur = self._extract_rank(user_query)
        rmin, rmax = data['rank_range']
        flat = []
        for inst, plist in data['institutes'].items():
            for p in plist:
                flat.append({
                    "inst": inst, "prog": p['program'], "cr": p['closing_rank'],
                    "rn": p['rank_num'], "bucket": self._bucket(ur, p['rank_num']),
                    "rd": p['round'], "yr": p['year']
                })

        def sort_key(x):
            order = {"Likely":0, "Balanced":1, "Reach":2, "":3}
            margin = (x['rn'] - ur) if (ur and x['rn']) else 10**9
            round_num = int(x['rd']) if str(x['rd']).isdigit() else -1
            return (order.get(x['bucket'],3), abs(margin), -round_num, x['rn'] or 10**9)

        flat = sorted(flat, key=sort_key)

        lines = []
        lines.append("## 🎓 Your JoSAA Options — Chat Summary")
        chips = []
        if ur: chips.append(f"Rank **{ur:,}**")
        chips.append("Assumed **OPEN · Gender-Neutral · AI · Final round**")
        lines.append(" • ".join(chips))
        note = rank_note_for_query(user_query)
        if note:
            lines.append("")
            lines.append(f"_Note: {note}_")
        lines.append("")
        lines.append(f"Found **{data['total']}** matching seats across **{len(data['institutes'])}** institutes.")
        if rmin and rmax: lines.append(f"Closing ranks range roughly **{rmin:,} → {rmax:,}**.")

        if flat:
            lines.append("")
            lines.append("### ⭐ Top matches for you")
            for x in flat[:6]:
                tag = f" — *{x['bucket']}*" if x['bucket'] else ""
                lines.append(f"- **{x['inst']}**, {x['prog']} (closing rank **{x['cr']}**, R{x['rd']}){tag}")

        # Strategy
        lines.append("")
        lines.append("### 🧭 Strategy")
        if ur and rmin:
            if ur <= rmin + 500:
                lines.append("- You’re **well inside** several cutoffs — prioritise campus/branch fit.")
            elif ur <= rmax:
                lines.append("- You’re in a **competitive** band — order carefully and track later rounds.")
            else:
                lines.append("- It’s **tight** — add more backups and watch special/spot rounds.")
        lines.append("- Build a 3–4–3 stack: **3 Reach • 4–6 Balanced • 3–4 Likely**.")
        lines.append("- Compare curriculum, location, internships/placements from official pages & alumni.")
        lines.append("- Round-to-round swings of ±300–600 are common in popular branches.")

        return "\n".join(lines)

    def _refine_final_answer(self, user_query: str, base_answer: str, rows: List[Dict], web_results: List[Dict]) -> str:
        row_lines = []
        for row in rows[:8]:
            closing_rank = row.get('closing_rank')
            if closing_rank in (None, ""):
                closing_rank = row.get('closing_rank_num', '')
            row_lines.append(
                f"- Institute: {row.get('institute', 'Unknown')} | Program: {row.get('program', 'Unknown')} | "
                f"Closing rank: {closing_rank} | Category: {row.get('category', '')} | "
                f"Quota: {row.get('quota', '')} | Gender: {row.get('gender', '')} | Round: {row.get('round', '')}"
            )

        web_lines = []
        for item in web_results[:8]:
            web_lines.append(
                f"- Title: {item.get('title', '')} | URL: {item.get('url', '')} | "
                f"{item.get('content', '')[:280]}"
            )

        prompt = f"""
You are helping with JoSAA counselling.

Rules:
- Admission feasibility must be based on the structured JoSAA rows only.
- The draft answer may come from a fine-tuned local model. Use it as a draft only, not as final truth.
- Use Gemini judgement to polish the draft into a clean final answer.
- Use the web results only to enrich with official/recent context such as placements, fees, curriculum, campus, or reputation.
- Do not invent facts not present in the inputs.
- Prefer concise markdown paragraphs and bullets.
- If web results are present, include a short "Web context" section. If not, omit it.
- Cite sources inline using the provided URLs in parentheses when using web facts.
- Do not output raw field names like closing_rank=, HTML entities, or escaped markdown.
- Preserve the user's actual exam/rank context and do not generalize away important constraints.

User query:
{user_query}

Structured JoSAA shortlist:
{chr(10).join(row_lines)}

Base answer draft:
{base_answer}

Web search results:
{chr(10).join(web_lines)}
"""
        try:
            res = self.sim.generate_content(prompt)
            text = (res.text or "").strip()
            return text or self._append_web_sources(base_answer, web_results)
        except Exception as e:
            log_error("Answer generation with web", e)
            return self._append_web_sources(base_answer, web_results)

    def _append_web_sources(self, base_answer: str, web_results: List[Dict]) -> str:
        if not web_results:
            return base_answer
        lines = [base_answer, "", "### Web context"]
        for item in web_results[:5]:
            title = item.get("title") or item.get("domain") or item.get("url")
            lines.append(f"- **{title}**: {item.get('url', '')}")
        return "\n".join(lines)

# ========================
# ORCHESTRATOR
# ========================
class Pipeline:
    def __init__(self):
        logger.info("🚀 Initializing JoSAA RAG Pipeline")
        start = time.time()
        self.pg = Pg(DB)
        self.schema_text = self.pg.fetch_schema_text()
        self.policy = Policy()
        self.enhancer = QueryEnhancer(GEMINI_API_KEY, self.policy)
        self.sqlpro = SqlGenPro(GEMINI_API_KEY, self.schema_text, self.policy)
        self.summary_drafter = HFSummaryDraftModel(
            enabled=USE_HF_SUMMARY,
            model_id=HF_MODEL_ID,
            token=HF_TOKEN,
            local_path=HF_MODEL_PATH,
        )
        self.answerer = Answerer(GEMINI_API_KEY, self.summary_drafter)
        self.search = TavilySearch(TAVILY_API_KEY)
        log_performance("Pipeline initialization", start)
        logger.info("🎉 JoSAA RAG Pipeline initialized successfully!")

    def _dedup_rows(self, rows: List[Dict]) -> List[Dict]:
        start = time.time()
        rows = sorted(rows, key=_sort_tuple)
        seen, out = set(), []
        for r in rows:
            k = _key_tuple(r)
            if k in seen: continue
            seen.add(k); out.append(r)
        log_performance("Row deduplication", start)
        return out

    @staticmethod
    def _pick_best(cands: List[Tuple[str, List[Dict]]]) -> Tuple[Optional[str], List[Dict]]:
        ne = [c for c in cands if len(c[1]) > 0]
        return (sorted(ne, key=lambda x: len(x[1]), reverse=True)[0]
                if ne else (cands[0] if cands else (None, [])))

    def _fallback_web(self, user_query: str, prefix: str = "", exam_kind: str = "") -> Tuple[str, str, List[Dict], List[Dict]]:
        if exam_kind:
            web_results = self.search.search_exam_guidance(user_query, exam_kind)
        else:
            web_results = self.search.search_general(user_query)
        answer = self.answerer.answer_from_web_only(user_query, web_results, exam_kind=exam_kind)
        if prefix:
            answer = f"{prefix}\n\n{answer}".strip()
        return answer, "", [], web_results

    def run(self, user_query: str) -> Tuple[str, str, List[Dict], List[Dict]]:
        logger.info(f"🚀 Starting pipeline execution for query: '{user_query}'")
        start = time.time()
        try:
            web_only_exam = detect_web_only_exam(user_query)
            if web_only_exam:
                exam_label = web_only_exam_label(web_only_exam)
                if self.search.enabled:
                    prefix = (
                        f"Your query is about **{exam_label}**. "
                        "This app's Supabase/Postgres data contains only **JoSAA** ranks, so I did not search the database for this answer. "
                        "I used **Tavily web search** on counselling-related websites instead, because these ranks are not directly comparable to JoSAA/JEE cutoffs."
                    )
                else:
                    prefix = (
                        f"Your query is about **{exam_label}**. "
                        "This app's Supabase/Postgres data contains only **JoSAA** ranks, so it should not use the database for this answer. "
                        "**Tavily web search is not configured right now**, so I can only return a limited fallback response."
                    )
                return self._fallback_web(user_query, prefix, exam_kind=web_only_exam)
            warning = deterministic_exam_warning(user_query)
            if warning:
                return self._fallback_web(user_query, warning)
            variants = self.enhancer.enhance(user_query) or [user_query]
            variants = [re.sub(r'^\s*[-*•]+\s*', '', v) for v in variants] or [user_query]

            sqls = []
            for v in variants:
                s = self.sqlpro.to_sql(v)
                if not s: continue
                try:
                    sqls.append(sanitize_select(s, 15))
                except Exception:
                    continue

            if not sqls:
                return self._fallback_web(
                    user_query,
                    "I couldn't produce a safe SQL for that query, so I used web search instead."
                )

            # unique sqls
            u, seen = [], set()
            for s in sqls:
                h = hash(s.strip().lower())
                if h in seen: continue
                seen.add(h); u.append(s)
            sqls = u

            # Execute
            cands = []
            for s in sqls:
                try:
                    rows = self.pg.run(s)
                    cands.append((s, rows))
                except Exception:
                    pass
            if not cands:
                return self._fallback_web(
                    user_query,
                    "The generated database queries failed to run, so I used web search instead."
                )

            best_sql, best_rows = self._pick_best(cands)
            best_rows = self._dedup_rows(best_rows)
            if not best_rows:
                return self._fallback_web(
                    user_query,
                    "The database did not return a useful shortlist for that query, so I used web search instead."
                )
            web_results = self.search.enrich(user_query, best_rows)
            answer = self.answerer.answer(user_query, best_rows, web_results)
            total = log_performance("Complete pipeline execution", start)
            logger.info(f"[Pipeline] rows={len(best_rows)} total={total:.1f}s")
            return answer, best_sql, best_rows, web_results
        except Exception as e:
            log_error("Pipeline execution", e)
            return self._fallback_web(
                user_query,
                f"The structured pipeline hit an internal error ({e}), so I used web search instead."
            )

# ========================
# STREAMLIT UI
# ========================
@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    return Pipeline()


def init_session_state():
    if 'pipeline' not in st.session_state:
        try:
            st.session_state.pipeline = get_cached_pipeline()
            st.session_state.pipeline_loaded = True
            logger.info("✅ Pipeline loaded in session state")
        except Exception as e:
            st.session_state.pipeline_loaded = False
            st.session_state.pipeline_error = str(e)
            log_error("Session state pipeline initialization", e)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None

def get_logs_tail():
    logs = st.session_state.log_stream.getvalue() if hasattr(st.session_state, 'log_stream') else ""
    return logs.split("\n")[-60:] if logs else []


def render_assistant_extras(message: Dict) -> None:
    if message.get("sql"):
        with st.expander("SQL used", expanded=False):
            st.code(message["sql"], language="sql")
    if message.get("data"):
        with st.expander("Matched rows", expanded=False):
            st.dataframe(pd.DataFrame(message["data"]), use_container_width=True)
    if message.get("web"):
        with st.expander("Web sources", expanded=False):
            for item in message["web"]:
                title = item.get("title") or item.get("url")
                st.markdown(f"- [{title}]({item.get('url', '')})")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                render_assistant_extras(message)


def run_query(query: str) -> None:
    logger.info(f"🎯 User submitted query: '{query}'")
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    st.session_state.last_error = None

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Working - understanding -> SQL -> database -> web context -> final summary"):
                answer, sql, rows, web_results = st.session_state.pipeline.run(query)
            st.markdown(answer)
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sql": sql,
                "data": rows,
                "web": web_results,
            }
            render_assistant_extras(assistant_message)
            st.session_state.messages.append(assistant_message)
            st.session_state.query_count += 1
        except Exception as e:
            log_error("UI query handling", e)
            error_text = f"Something failed while processing the query: {e}"
            st.error(error_text)
            st.session_state.last_error = error_text
            st.session_state.messages.append({"role": "assistant", "content": error_text})

def main():
    st.set_page_config(page_title="JoSAA AI Assistant", page_icon="🎓", layout="wide")

    st.markdown("""
    <style>
    .bubble { padding: 1rem 1.2rem; border-radius: 14px; margin: 0.8rem 0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06); line-height: 1.55; font-size: 0.98rem; }
    .user { background: #e8f0fe; border-left: 5px solid #1a73e8; }
    .assistant { background: #eef7ee; border-left: 5px solid #34a853; }
    .header { text-align:center; padding: 1rem; background: linear-gradient(45deg,#667eea,#764ba2);
      color:#fff; border-radius: 12px; margin-bottom: 0.5rem; }
    .log { background:#0f172a; color:#e2e8f0; padding:.6rem; border-radius:10px; font:12px ui-monospace; max-height:300px; overflow:auto;}
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    st.markdown("<div class='header'><h3 style='margin:0'>🎓 JoSAA Counselling Assistant</h3><div>JoSAA database answers with Tavily web fallback.</div></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("🔧 Status")
        if st.session_state.get('pipeline_loaded', False):
            st.success("Pipeline & DB ready")
        else:
            st.error("Pipeline failed")
            st.write(st.session_state.get('pipeline_error',''))

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.last_error = None
            st.rerun()

        st.subheader("📝 Logs (tail)")
        st.markdown(f"<div class='log'>{'<br>'.join(get_logs_tail())}</div>", unsafe_allow_html=True)

        st.subheader("💡 Try one")
        samples = [
            "Which colleges are good for Mechanical around AIR 7500?",
            "Electrical engineering options near rank 8000",
            "I have AIR 6000, which IIT programs can I get?",
            "Which college will I get if my rank is 5000?",
            "I got 5100 rank in AP EAMCET, suggest me some best colleges"
        ]
        for q in samples:
            if st.button(q, use_container_width=True, key=f"s_{hash(q)}"):
                st.session_state.pending_query = q

    st.subheader("💬 Chat")
    if st.session_state.get("last_error"):
        st.warning(st.session_state["last_error"])

    render_chat_history()

    queued_query = st.session_state.pending_query
    if queued_query:
        st.session_state.pending_query = None
        if st.session_state.get('pipeline_loaded', False):
            run_query(queued_query)

    prompt = st.chat_input(
        "Ask about JoSAA cutoffs college suggestions and counselling questions...",
        disabled=not st.session_state.get('pipeline_loaded', False),
    )
    if prompt and prompt.strip():
        run_query(prompt.strip())

if __name__ == "__main__":
    main()
