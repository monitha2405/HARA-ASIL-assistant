"""Local LLM through Ollama (free, runs on your machine). Default model: qwen2.5:0.5b (~400 MB).
If Ollama is not running we return a clearly labelled template draft instead, so the app never breaks."""
import os, json, urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")

def build_prompt(desc, s, e, c, asil, similar):
    ctx = "\n".join(f"- [{x['id']}] {x['text']} (S{x['S']} E{x['E']} C{x['C']} -> ASIL {x['ASIL']})" for x in similar)
    return ("You are a functional safety assistant. Use ONLY the context below. Do not change the ASIL.\n"
            f"Context (approved similar hazards):\n{ctx}\n\n"
            f"New hazard: {desc}\nClassified S{s} E{e} C{c}, ASIL {asil}.\n"
            "Write: 1) one safety goal sentence, 2) one functional safety requirement sentence, "
            "3) cite the context ids you used. Keep it under 80 words.")

def draft(desc, s, e, c, asil, similar):
    try:
        body = json.dumps({"model": MODEL, "prompt": build_prompt(desc, s, e, c, asil, similar),
                           "stream": False, "options": {"temperature": 0.2, "num_predict": 160}}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/generate", body, {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=60))["response"].strip()
        return {"source": f"LLM ({MODEL})", "text": text}
    except Exception:
        ids = ", ".join(x["id"] for x in similar)
        return {"source": "template (LLM offline)",
                "text": f"Safety goal: avoid the hazard '{desc}' at ASIL {asil}. "
                        f"FSR: the item shall detect and mitigate this malfunction to reach a safe state. Based on {ids}."}
