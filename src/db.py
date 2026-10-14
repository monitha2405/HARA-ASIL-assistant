"""PostgreSQL structured storage: assessments + engineer decisions (audit trail)."""
import os, json
import psycopg2, psycopg2.extras

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/hara")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    hazard_id     TEXT PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT now(),
    description   TEXT NOT NULL,
    inputs        JSONB NOT NULL,
    s INT, e INT, c INT,
    asil          TEXT NOT NULL,
    min_confidence REAL,
    needs_review  BOOLEAN,
    draft_source  TEXT,
    draft_text    TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    hazard_id   TEXT REFERENCES assessments(hazard_id),
    action      TEXT NOT NULL CHECK (action IN ('ACCEPT','REJECT')),
    engineer    TEXT,
    note        TEXT,
    asil_final  TEXT
);
"""

def conn():
    return psycopg2.connect(DB_URL)

def init_db():
    with conn() as c, c.cursor() as cur:
        cur.execute(SCHEMA)

def save_assessment(r, inputs, draft):
    with conn() as c, c.cursor() as cur:
        cur.execute("""INSERT INTO assessments(hazard_id,description,inputs,s,e,c,asil,min_confidence,needs_review,draft_source,draft_text)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (r["hazard_id"], r["description"], json.dumps(inputs), r["S"]["value"], r["E"]["value"],
                     r["C"]["value"], r["asil"], r["min_confidence"], r["needs_expert_review"], draft["source"], draft["text"]))

def save_decision(d):
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO decisions(hazard_id,action,engineer,note,asil_final) VALUES (%s,%s,%s,%s,%s)",
                    (d["hazard_id"], d["action"], d.get("engineer", ""), d.get("note", ""), d.get("asil_final", "")))

def recent_audit(limit=50):
    """Assessments and decisions merged into one timeline, newest first."""
    with conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT created_at AS ts, 'assess' AS event, hazard_id,
                              description || ' -> ASIL ' || asil AS details FROM assessments
                       UNION ALL
                       SELECT created_at, 'decision', hazard_id,
                              action || ' by ' || COALESCE(NULLIF(engineer,''),'-') || ' ' || COALESCE(note,'') FROM decisions
                       ORDER BY ts DESC LIMIT %s""", (limit,))
        rows = cur.fetchall()
    for r in rows: r["ts"] = r["ts"].strftime("%Y-%m-%d %H:%M:%S")
    return rows
