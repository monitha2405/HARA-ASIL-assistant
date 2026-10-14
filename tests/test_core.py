import pytest
from src.asil import asil
from src.data_gen import generate

def test_asil_extremes():
    assert asil(3,4,3)=="D" and asil(1,1,1)=="QM" and asil(0,4,3)=="QM" and asil(3,3,3)=="C" and asil(2,3,3)=="B" and asil(2,2,3)=="A"

def test_dataset_shape():
    d = generate(200); assert len(d)==200 and set(d.ASIL) <= {"QM","A","B","C","D"}

def test_llm_fallback_when_offline(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "OLLAMA_URL", "http://localhost:1")   # nothing listens here
    out = llm.draft("loss of braking", 3, 4, 3, "D", [{"id":"LIB-1"}])
    assert out["source"].startswith("template")

def test_rag_returns_similar():
    from src import rag
    r = rag.similar("loss of emergency braking while highway cruising", 3)
    assert len(r) == 3 and "ASIL" in r[0]

def _db_up():
    try:
        from src import db; db.conn().close(); return True
    except Exception: return False

@pytest.mark.skipif(not _db_up(), reason="PostgreSQL not running")
def test_api_end_to_end():
    from app.main import app
    c = app.test_client()
    ok = c.post("/api/assess", json={"function":"emergency braking","malfunction":"loss of","scenario":"highway cruising","speed_kmh":110,"exposure_pct":25})
    j = ok.get_json(); assert ok.status_code == 200 and j["asil"] in {"QM","A","B","C","D"} and len(j["similar"]) == 3
    d = c.post("/api/decision", json={"hazard_id": j["hazard_id"], "action": "ACCEPT", "engineer": "test"})
    assert d.status_code == 200
    assert any(r["hazard_id"] == j["hazard_id"] for r in c.get("/api/audit").get_json())
    assert c.post("/api/assess", json={"function":"nonsense"}).status_code == 400
