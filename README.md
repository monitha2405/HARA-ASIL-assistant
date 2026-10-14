# HARA ASIL Assistant
Hazard Analysis & Risk Assessment helper: ML predicts Severity / Exposure / Controllability, ASIL is computed by a
deterministic rule, a **RAG** layer retrieves similar approved hazards, a **local LLM** drafts a safety goal + requirement,
and everything is stored in **PostgreSQL** with human Accept/Reject.

## Tech stack (from the case-study document; "or" choices -> the easiest one)
| Layer | Case study options | Used here |
|---|---|---|
| UI | Flask-served web UI |
| API | Flask | 
| LLM |  **Qwen2.5-0.5B via Ollama** |
| Embeddings | MiniLM-L6 |
| Vector DB | ChromaDB |
| RAG | custom Python | 
| Structured storage | PostgreSQL |
| Deployment | Docker | 

## Run with Docker (easiest, everything included)
Install Docker Desktop and make sure it is running. Then, in the project folder (where `docker-compose.yml` is):
```
docker compose up --build
```
The first run takes 5-10 minutes: it builds the app, trains the model, and downloads the 400 MB LLM automatically
(`model-pull` service). When the app reports it is running on port 8000, open http://localhost:8000

| Task | Command |
|---|---|
| Start again | `docker compose up` |
| Stop | `Ctrl+C` or `docker compose down` |
| Stop and delete stored data | `docker compose down -v` |

Docker uses its own PostgreSQL, so stop any locally running copy of the app first (both use port 8000).

## Run without Docker
**Prerequisites:** Python 3.10+, PostgreSQL, and (optional) Ollama. Without Ollama the app still works but shows a
clearly labelled template draft instead of an LLM draft.

**1. Create and activate a virtual environment, install packages**
```
python -m venv venv
venv\Scripts\activate                # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
**2. Create the database**
```
psql -U postgres -c "CREATE DATABASE hara;"
```
**3. Set the database URL** (only if your PostgreSQL password is not `postgres`; repeat in every new terminal)
```
$env:DATABASE_URL="postgresql://postgres:YOURPASSWORD@localhost:5432/hara"     # PowerShell
set DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/hara        # CMD
export DATABASE_URL="postgresql://postgres:YOURPASSWORD@localhost:5432/hara"   # Mac/Linux
```
**4. Download the LLM** (optional, ~400 MB; keep Ollama running)
```
ollama pull qwen2.5:0.5b
ollama list
```
**5. Train the models and build the knowledge base** (run once; needs internet for the embedding model)
```
python -m src.train
```
**6. Start the app**
```
python -m app.main
```
Open http://localhost:8000

**7. Run the tests** (PostgreSQL must be running)
```
pytest -q
```

## Pipeline
`data_gen.py` (synthetic data) → `train.py` (EDA, imputation, TF-IDF+SVD, SelectKBest, 4 models, CV, GridSearch, test;
saves `models/hara_model.skops`) → `asil.py` (rule) → `rag.py` (ChromaDB retrieval) → `llm.py` (Ollama draft) →
`db.py` (PostgreSQL audit) → Flask UI.

## Rubric mapping
Problem/approach: this README · Dataset/EDA: `data_gen.py`, `reports/eda_*.png` · Features/SVD/selection: `train.py` ·
Models (18): LogReg/SVM/RF/kNN + baseline · Evaluation: 5-fold CV, GridSearch, confusion matrix, under-classification rate ·
Demo: Flask UI (assess / performance / audit tabs) · Code quality: `src/`, `app/`, `tests/` · MLOps: Docker, compose, `/health`, versioned `metrics.json`, PostgreSQL audit ·
Responsible AI: below.

## Responsible AI & limitations
- Data is **synthetic**; the "approved library" in ChromaDB is built from it. Replace with real approved HARA data.
- The LLM can never change the ASIL (it is given as fixed input); it only drafts text from retrieved context and cites IDs.
  A 0.5B model is weak: always review drafts. Swap `OLLAMA_MODEL` for a bigger model (e.g. `llama3.2:3b`) if you have the RAM.
- ASIL C/D or low confidence → mandatory expert review. Every assessment and decision is stored in PostgreSQL.
- Verify the ASIL table against your licensed ISO 26262-3 copy.

## Troubleshooting
- `ollama` not recognized in VS Code: restart VS Code, or run it from CMD.
- Draft says "template (LLM offline)": Ollama is not running or the model is not downloaded.
- `password authentication failed`: wrong password in `DATABASE_URL`.
- `database "hara" does not exist`: run step 2 again.
- `collection approved_hazards does not exist`: run `python -m src.train` first.
- Docker `cannot connect`: Docker Desktop is not running.
