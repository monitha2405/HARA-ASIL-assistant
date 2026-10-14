"""Flask web app: serves the UI and a small JSON API."""
import json, time, pickle
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from src.data_gen import FUNCS, MALS, SCEN, compose
from src.asil import asil
from src import db, rag, llm

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=None)
db.init_db()   # create tables if they do not exist yet
MODEL = pickle.load(open(ROOT / "models" / "hara_model.pkl", "rb"))   # dict: S, E, C pipelines

# allowed range for each numeric input (simple manual validation)
LIMITS = {"speed_kmh": (0, 250), "pedestrians": (0, 1), "exposure_pct": (0.001, 100),
          "reaction_s": (0, 10), "occupants": (1, 8)}
DEFAULTS = {"speed_kmh": 50, "pedestrians": 0, "exposure_pct": 5, "reaction_s": 1.5, "occupants": 2}

def parse(body):
    """Validate request JSON; returns (clean_dict, error_message)."""
    for key, allowed in (("function", FUNCS), ("malfunction", MALS), ("scenario", SCEN)):
        if body.get(key) not in allowed:
            return None, f"invalid {key}"
    clean = {k: body.get(k) for k in ("function", "malfunction", "scenario")}
    for k, (lo, hi) in LIMITS.items():
        try:
            v = float(body.get(k, DEFAULTS[k]))
        except (TypeError, ValueError):
            return None, f"{k} must be a number"
        if not lo <= v <= hi:
            return None, f"{k} must be between {lo} and {hi}"
        clean[k] = v
    return clean, None

@app.get("/")
def index(): return send_from_directory(ROOT / "app" / "static", "index.html")
@app.get("/reports/<path:f>")
def reports(f): return send_from_directory(ROOT / "reports", f)
@app.get("/health")
def health():
    meta = json.load(open(ROOT / "models" / "metrics.json"))["meta"]
    return jsonify(status="ok", model_version=meta["version"])
@app.get("/api/options")
def options(): return jsonify(functions=list(FUNCS), malfunctions=list(MALS), scenarios=list(SCEN))
@app.get("/api/metrics")
def metrics(): return jsonify(json.load(open(ROOT / "models" / "metrics.json")))

@app.post("/api/assess")
def assess():
    h, err = parse(request.get_json(silent=True) or {})
    if err: return jsonify(error=err), 400
    text = compose(h["function"], h["malfunction"], h["scenario"], 0)
    X = pd.DataFrame([{"text": text, **{k: h[k] for k in LIMITS}}])
    res = {}
    for t, p in MODEL.items():
        pr = p.predict_proba(X)[0]; i = int(pr.argmax())
        res[t] = {"value": int(p.classes_[i]), "confidence": round(float(pr[i]), 3),
                  "probs": {str(c): round(float(v), 3) for c, v in zip(p.classes_, pr)}}
    a = asil(res["S"]["value"], res["E"]["value"], res["C"]["value"])
    conf = min(v["confidence"] for v in res.values())
    review = conf < .6 or a in ("C", "D")
    out = {"hazard_id": f"HZ-{int(time.time()*1000)}", "description": text, **res, "asil": a,
           "min_confidence": conf, "needs_expert_review": review,
           "rationale": f"S{res['S']['value']}+E{res['E']['value']}+C{res['C']['value']} -> ASIL {a} via deterministic rule; "
                        f"{'flagged for mandatory expert review (low confidence or high ASIL)' if review else 'routine review'}. AI suggestion only."}
    out["similar"] = rag.similar(text, 3)
    out["draft"] = llm.draft(text, res["S"]["value"], res["E"]["value"], res["C"]["value"], a, out["similar"])
    db.save_assessment(out, h, out["draft"])
    return jsonify(out)

@app.post("/api/decision")
def decision():
    d = request.get_json(silent=True) or {}
    if d.get("action") not in ("ACCEPT", "REJECT") or not d.get("hazard_id"):
        return jsonify(error="action must be ACCEPT or REJECT, hazard_id required"), 400
    db.save_decision(d)
    return jsonify(stored=True)

@app.get("/api/audit")
def audit():
    """Last 50 events from PostgreSQL, newest first."""
    return jsonify(db.recent_audit(50))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
