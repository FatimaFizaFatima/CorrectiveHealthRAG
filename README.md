# CorrectiveHealthRAG

A FastAPI-backed Agentic RAG assistant that answers questions about **depression and hypertension** using official health documents from **NIMH and WHO**.

Built with **LangGraph** using the Corrective RAG pattern, served through **FastAPI**, with query history logged to **PostgreSQL**.

## How It Works

```text
Question
   ↓
Decide (call retriever or answer directly)
   ↓
Retrieve Documents
   ↓
Check Relevance
   ↓
Rewrite if Needed (loops back to Retrieve)
   ↓
Generate Answer
   ↓
Log to PostgreSQL
```

The system:

* Decides whether the question needs document retrieval or is off-topic
* Retrieves relevant chunks from the source PDFs
* Grades whether the retrieved content actually answers the question
* Rewrites the question and retries (up to a set limit) if retrieval was poor
* Generates a final answer using only the retrieved context
* Logs every question, answer, and source filename(s) to PostgreSQL

## Tech Stack

* **LangGraph** — agentic RAG orchestration (Corrective RAG pattern)
* **FastAPI** — HTTP API layer
* **Gemini 3.1 Flash Lite** — generation and rewriting
* **HuggingFace `all-MiniLM-L6-v2`** — embeddings
* **InMemoryVectorStore** — vector storage (rebuilt from source PDFs on startup)
* **PostgreSQL + SQLAlchemy + Alembic** — query history logging and migrations
* **LangChain** — retriever tooling

## Project Structure

```text
CorrectiveHealthRAG/
│
├── app/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── config.py                # LLM client setup
│   ├── settings.py              # Environment variable config (pydantic-settings)
│   │
│   ├── router/
│   │   └── corrective_health_router.py   # /corrective-health routes
│   │
│   ├── schema/
│   │   └── corrective_health_schema.py   # Pydantic request/response models
│   │
│   ├── model/
│   │   └── corrective_health_model.py    # SQLAlchemy CorrectiveRAGModel (query log table)
│   │
│   ├── db/
│   │   ├── database.py          # Engine, SessionLocal, Base
│   │   └── session.py           # get_db dependency
│   │
│   └── src/
│       ├── ingest.py
│       ├── retriever.py
│       ├── state.py
│       ├── logger.py
│       ├── graph.py
│       ├── utils/
│       │   └── extract_content.py        # Shared response-parsing helpers
│       └── node/
│           ├── decide.py
│           ├── grade.py
│           ├── rewrite.py
│           └── generate.py
│
├── alembic/                     # Database migrations
│   └── versions/
├── alembic.ini
│
├── data/
│   └── source_docs/             # depression.pdf, hypertension.pdf
│
├── tests/
│   └── test_queries.py
│
├── requirements.txt
└── .env                         # not committed — see Setup
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY
DATABASE_URL
```

Add the required PDFs to:

```text
data/source_docs/
```

Run database migrations:

```bash
alembic upgrade head
```

## Running the App

Start the API server:

```bash
uvicorn app.main:app --reload
```

Interactive API docs available at:

```text
http://127.0.0.1:8000/docs
```

### API Endpoint

**POST** `/corrective-health/`

Request:
```json
{
  "question": "What are the symptoms of depression?"
}
```

Response:
```json
{
  "answer": "Common signs and symptoms of depression include..."
}
```

Each call is logged to the `corrective_rags` table in PostgreSQL, including the question, answer, source filename(s), and timestamp.

### Standalone graph testing (no API/DB)

```bash
python -m tests.test_queries
```

## Deployment

This project is being deployed on AWS EC2 (Ubuntu) with PostgreSQL, Gunicorn, and Nginx. Deployment steps are documented separately.