# OSS Contribution Agent

An AI-powered agent that helps developers contribute to open source projects. Give it your skills and a GitHub repo — it finds the right issues, explains the codebase, and guides you through your first contribution.

Built as a production LLM engineering system demonstrating the full stack: RAG, multi-agent orchestration, async streaming, LLMOps tracing, and GCP deployment.

---

## What It Does

1. **Skill Matching** — Input your tech stack, get ranked OSS repos that match your skills
2. **Issue Analysis** — Scores open issues by difficulty and skill match, recommends top 2–3
3. **Issue Deep-Dive** — Reads the full issue, helps you reproduce it, points to the exact file/folder
4. **Codebase Navigation** — RAG over actual source files, explains relevant code, suggests where the fix goes
5. **Guided Contribution** — Multi-turn conversational agent walks you through the full contribution flow

---

## Architecture

```
User Query
    │
    ▼
FastAPI (Async + Streaming)
    │
    ├──► GitHub Fetcher (PyGithub)
    │         └── README, CONTRIBUTING, file structure, good-first-issues
    │
    ├──► Embedder (OpenAI text-embedding-3-small → ChromaDB)
    │         └── Sliding window chunking (500 words, 50 overlap)
    │
    ├──► Retriever
    │         ├── Stage 1: ChromaDB semantic search (top_k=10)
    │         └── Stage 2: Cohere rerank-v3.5 (top_n=3)
    │
    └──► Claude Sonnet (claude-sonnet-4-6)
              └── Streaming response → Streamlit UI

Observability: LangSmith traces every stage (fetch → embed → retrieve → rerank → generate)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet (claude-sonnet-4-6) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB (persistent) |
| Reranking | Cohere rerank-v3.5 |
| API | FastAPI (async + streaming) |
| Frontend | Streamlit |
| Tracing | LangSmith |
| GitHub | PyGithub |
| Deployment | Docker + GCP Cloud Run |

---

## Project Structure

```
oss-rag/
├── src/
│   ├── ingestion/
│   │   └── github_fetcher.py      # Fetches README, CONTRIBUTING, issues via GitHub API
│   ├── embeddings/
│   │   └── embedder.py            # Chunks text, generates embeddings, stores in ChromaDB
│   ├── retrieval/
│   │   └── retriever.py           # Two-stage retrieval: semantic search + Cohere reranking
│   ├── api/
│   │   └── main.py                # FastAPI endpoints: /query (sync) and /query/stream (async)
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS evaluation: faithfulness, relevancy, precision, recall
│   └── app.py                     # Streamlit frontend
├── requirements.txt
├── setup.py
└── .env                           # API keys (never committed)
```

---

## Quickstart

**Prerequisites:** Python 3.10+, API keys for Anthropic, OpenAI, Cohere, GitHub

```bash
# Clone and set up
git clone https://github.com/C3Suryansu/oss-rag
cd oss-rag
python -m venv venv-oss-rag
source venv-oss-rag/bin/activate
pip install -e .

# Add your API keys
cp .env.example .env
# Edit .env with your keys

# Start the API server
uvicorn src.api.main:app --reload

# Start the frontend (new terminal)
streamlit run src/app.py
```

---

## API

### POST /query
Synchronous endpoint. Returns full JSON response with answer and sources.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/kubeflow/pipelines", "question": "How do I set up the dev environment?"}'
```

### POST /query/stream
Streaming endpoint. Returns tokens as they're generated.

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/kubeflow/pipelines", "question": "How do I set up the dev environment?"}' \
  --no-buffer
```

### GET /health
```bash
curl http://localhost:8000/health
```

---

## LLMOps: LangSmith Tracing

Every pipeline stage is instrumented with LangSmith:

| Span | What it captures |
|------|-----------------|
| `oss_rag_query` | Full pipeline — top-level trace |
| `fetch_repo_data` | GitHub API call latency + fetched data |
| `retrieve_and_rerank` | Full retrieval pipeline |
| `chromadb_retrieve` | Semantic search results + distances |
| `get_query_embedding` | OpenAI embedding call latency |
| `cohere_rerank` | Reranking scores + final ordering |
| `claude_generate` | Claude call — prompt in, answer out |

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable.

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
COHERE_API_KEY=
GITHUB_PAT=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=oss-rag
```

---

## Roadmap (WIP)

- [ ] CrewAI skill-matching engine
- [ ] LangGraph conversational agent
- [ ] MCP server for GitHub tooling
- [ ] Qdrant migration + vector DB comparison
- [ ] RAGAS evaluation scores
- [ ] Fine-tuned issue difficulty classifier (LoRA)
- [ ] GCP Cloud Run deployment

---

## Author

**Suryansu Dash** — LLM Engineer & AI Consultant  
Building production AI systems for US startups.  
[GitHub](https://github.com/C3Suryansu) · [LinkedIn](https://linkedin.com/in/suryansu-dash)