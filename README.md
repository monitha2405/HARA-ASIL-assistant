# HARA ASIL Assistant — Functional Safety AI (Case Study 2)
Hazard Analysis & Risk Assessment helper: ML predicts Severity / Exposure / Controllability, ASIL is computed by a
deterministic rule, a **RAG** layer retrieves similar approved hazards, a **local LLM** drafts a safety goal + requirement,
and everything is stored in **PostgreSQL** with human Accept/Reject.

## Tech stack (from the case-study document; "or" choices -> the easiest one)
| Layer | Case study options | Used here |
|---|---|---|
| UI | Streamlit / Flask web UI | Flask-served web UI |
| API | FastAPI / Flask | Flask |
| LLM | Llama, Mistral, Qwen... (local) | **Qwen2.5-0.5B via Ollama** (free, ~400 MB) |
| Embeddings | BGE / E5 / Sentence Transformers | MiniLM-L6 (Chroma default, ~80 MB) |
| Vector DB | FAISS / ChromaDB | **ChromaDB** |
| RAG | LangChain / custom Python | custom Python (`src/rag.py`, `src/llm.py`) |
| Structured storage | SQLite / PostgreSQL | **PostgreSQL** (`src/db.py`) |
| Deployment | Docker | Dockerfile + docker-compose |
| ML (course syllabus) | – | scikit-learn: LogReg, SVM, RandomForest, kNN |

## Run with Docker (easiest, everything included)
```
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:0.5b     # first time only
```
Open http://localhost:8000

## Run without Docker
1. Install PostgreSQL, create a database: `createdb hara` (default URL `postgresql://postgres:postgres@localhost:5432/hara`, override with `DATABASE_URL`).
2. Install Ollama (ollama.com), then `ollama pull qwen2.5:0.5b`  (optional: if Ollama is off the app shows a clearly labelled template draft instead).
3. ```
   pip install -r requirements.txt
   python -m src.train      # data, EDA plots, model training, builds ChromaDB library (run once)
   python -m app.main       # http://localhost:8000
   pytest -q
   ```

## Pipeline
`data_gen.py` (synthetic data) → `train.py` (EDA, imputation, TF-IDF+SVD, SelectKBest, 4 models, CV, GridSearch, test) →
`asil.py` (rule) → `rag.py` (ChromaDB retrieval) → `llm.py` (Ollama draft) → `db.py` (PostgreSQL audit) → Flask UI.

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
