# HARA ASIL Assistant
Hazard Analysis & Risk Assessment helper: ML predicts Severity / Exposure / Controllability, ASIL is computed by a
deterministic rule, a **RAG** layer retrieves similar approved hazards, a **local LLM** drafts a safety goal + requirement,
and everything is stored in **PostgreSQL** with human Accept/Reject.

## Tech stack 
| Layer | Tech Stack used |
|---|---|---|
| UI | Flask-served web UI |
| API | Flask | 
| LLM |  Qwen2.5-0.5B via Ollama |
| Embeddings | MiniLM-L6 |
| Vector DB | ChromaDB |
| RAG | custom Python | 
| Structured storage | PostgreSQL |
| Deployment | Docker | 

## Run with Docker 
Install Docker Desktop and make sure it is running. Then, in the project folder (where `docker-compose.yml` is):
```
docker compose up --build
```
it builds the app, trains the model, and downloads the 400 MB LLM automatically by `model-pull` service. When the app reports it is running on port 8000, open http://localhost:8000

| Task | Command |
|---|---|
| Start again | `docker compose up` |
| Stop | `Ctrl+C` or `docker compose down` |
| Stop and delete stored data | `docker compose down -v` |

Docker uses its own PostgreSQL, so stop any locally running copy of the app first (both use port 8000).

## Run without Docker
**Prerequisites:** Python 3.10+, PostgreSQL, and Ollama. Without Ollama the app still works but shows a
clearly labelled template draft instead of an LLM draft.

**1. Create and activate a virtual environment, install packages**
```            
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

**7. Run the tests** 
```
pytest -q
```

## Pipeline
`data_gen.py` (synthetic data) → `train.py` (EDA, imputation, TF-IDF+SVD, SelectKBest, 4 models, CV, GridSearch, test;
saves `models/hara_model.skops`) → `asil.py` (rule) → `rag.py` (ChromaDB retrieval) → `llm.py` (Ollama draft) →
`db.py` (PostgreSQL audit) → Flask UI.




