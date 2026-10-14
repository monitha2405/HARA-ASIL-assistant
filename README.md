# HARA ASIL Assistant

**Hazard Analysis & Risk Assessment Assistant**

An AI-assisted hazard analysis system that predicts **Severity, Exposure, and Controllability** using machine learning, calculates Automotive Safety Integrity Level (ASIL) through deterministic rules, retrieves similar approved hazards using Retrieval-Augmented Generation (RAG), and generates draft safety goals and requirements using a locally hosted LLM.

All hazard assessments are stored in PostgreSQL and support human **Accept/Reject** review.

---

## Key Features

- **ML-Based Risk Assessment:** Predicts Severity, Exposure, and Controllability.
- **Deterministic ASIL Calculation:** Computes ASIL using predefined rules.
- **RAG-Based Retrieval:** Retrieves similar approved hazards from a vector database.
- **AI-Assisted Drafting:** Generates draft safety goals and requirements using a local LLM.
- **Audit Trail:** Stores structured assessment data in PostgreSQL.
- **Human-in-the-Loop Review:** Allows reviewers to accept or reject generated results.
- **Containerized Deployment:** Supports Docker Compose for simplified setup.

---

## Technology Stack

| Component | Technology | Purpose |
|:--|:--|:--|
| Frontend / UI | Flask-served Web UI | User interaction and assessment review |
| Backend API | Flask | Application endpoints and orchestration |
| Machine Learning | Python, scikit-learn | Predicts hazard assessment parameters |
| ASIL Calculation | Deterministic Python rules | Calculates ASIL from assessment parameters |
| Large Language Model | Qwen2.5-0.5B via Ollama | Drafts safety goals and requirements |
| Embeddings | MiniLM-L6 | Converts hazard descriptions into vector embeddings |
| RAG Pipeline | Custom Python implementation | Retrieves similar approved hazards |
| Vector Database | ChromaDB | Stores and searches hazard embeddings |
| Structured Database | PostgreSQL | Stores assessments and review records |
| Deployment | Docker, Docker Compose | Builds and runs application services |
| Testing | pytest | Runs automated tests |

---

## System Architecture and Pipeline

The application follows this processing pipeline:

```text
                    ┌───────────────────────┐
                    │       User Input      │
                    │ Hazard Description    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Flask Web UI / API  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  ML Risk Prediction   │
                    │ Severity, Exposure,   │
                    │ Controllability       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Deterministic ASIL    │
                    │ Rule Calculation      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ RAG Retrieval         │
                    │ ChromaDB + MiniLM-L6  │
                    │ Similar Approved     │
                    │ Hazards               │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Local LLM via Ollama  │
                    │ Draft Safety Goal and │
                    │ Safety Requirement    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ PostgreSQL            │
                    │ Assessment and Audit  │
                    │ Records               │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Human Review          │
                    │ Accept / Reject       │
                    └───────────────────────┘
```

### Model Training Pipeline

The training workflow is separate from the live assessment workflow.

```text
Synthetic Data Generation
          │
          ▼
Exploratory Data Analysis (EDA)
          │
          ▼
Data Imputation and Preprocessing
          │
          ▼
TF-IDF + SVD Feature Extraction
          │
          ▼
SelectKBest Feature Selection
          │
          ▼
Train and Compare Four ML Models
          │
          ▼
Cross-Validation and GridSearch
          │
          ▼
Test-Set Evaluation
          │
          ▼
Save Trained Model
models/hara_model.skops
```

---

## Prerequisites

For Docker deployment:

- Docker Desktop
- Docker Compose
- Internet connection for initial image and model downloads

For running without Docker:

- Python 3.10 or later
- PostgreSQL
- Ollama (optional; required for LLM-generated drafts)
- Internet connection to download the embedding model

---

## Run with Docker

### 1. Start Docker Desktop

Install Docker Desktop and ensure the Docker engine is running.

### 2. Open the project directory

Navigate to the directory containing `docker-compose.yml`.

### 3. Build and start the application

```bash
docker compose up --build
```

This builds the application, runs the model-training workflow, and downloads the approximately 400 MB LLM through the `model-pull` service, as configured in Docker Compose.

### 4. Open the application

Once the application is running, open:

**http://localhost:8000**

### Docker Commands

| Task | Command |
|:--|:--|
| Start services | `docker compose up` |
| Rebuild and start | `docker compose up --build` |
| Run in background | `docker compose up -d` |
| View logs | `docker compose logs -f` |
| Stop services | `docker compose down` |
| Stop and delete stored volumes | `docker compose down -v` |

> **Warning:** `docker compose down -v` deletes data stored in Docker-managed volumes, including persisted database data where applicable.

---

## Run Without Docker

### Step 1: Create a virtual environment

From the project root, run:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Step 2: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Create the PostgreSQL database

Ensure PostgreSQL is running, then execute:

```bash
psql -U postgres -c "CREATE DATABASE hara;"
```

If the database already exists, PostgreSQL will report an error; do not recreate it unnecessarily.

### Step 4: Configure the database connection

The default database connection uses the local PostgreSQL instance. If your password differs from the configured default, set `DATABASE_URL`.

**PowerShell:**

```powershell
$env:DATABASE_URL="postgresql://postgres:YOURPASSWORD@localhost:5432/hara"
```

**Command Prompt:**

```bat
set DATABASE_URL=postgresql://postgres:YOURPASSWORD@localhost:5432/hara
```

Replace `YOURPASSWORD` with your PostgreSQL password. Set the variable again when opening a new terminal.

### Step 5: Install the local LLM (optional)

Install Ollama and keep its service running. Then execute:

```bash
ollama pull qwen2.5:0.5b
ollama list
```

Without Ollama, the application can use a clearly labelled template draft instead of an LLM-generated draft, provided the application is configured for that fallback.

### Step 6: Train the models and build the knowledge base

Run this once, or again whenever you need to regenerate the model and knowledge base:

```bash
python -m src.train
```

This step requires an internet connection if the embedding model has not already been downloaded.

### Step 7: Start the application

```bash
python -m app.main
```

Open **http://localhost:8000** in your browser.

### Step 8: Run automated tests

```bash
pytest -q
```

---

## Project Structure

The following is a logical overview of the main components. Actual paths should match the repository.

```text

HARA-ASIL-assistant/
├── app/
│   ├── static/
│   │   └── index.html
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── hara_synthetic.csv
│
├── models/
│   ├── hara_model.pkl
│   └── metrics.json
│
├── reports/
│   ├── confusion_asil.png
│   ├── eda_class_distribution.png
│   ├── eda_correlation.png
│   └── eda_pca.png
│
├── src/
│   ├── __init__.py
│   ├── asil.py
│   ├── data_gen.py
│   ├── db.py
│   ├── llm.py
│   ├── rag.py
│   └── train.py
│
├── tests/
│   └── test_core.py
│
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt

```
