import os
import anthropic
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.ingestion.github_fetcher import fetch_repo_data
from src.embeddings.embedder import embed_repo_data, chroma_client
from src.retrieval.retriever import retrieve_and_rerank

load_dotenv()

app = FastAPI(title="OSS Onboarding RAG")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class QueryRequest(BaseModel):
    repo_url: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    try:
        # Step 1: Fetch and embed repo data
        repo_name = request.repo_url.rstrip("/").split("/")[-2] + "__" + \
                    request.repo_url.rstrip("/").split("/")[-1]

        # Check if collection already exists, if not, fetch and embed
        try:
            chroma_client.get_collection(repo_name)
            print(f"Collection {repo_name} already exists, skipping fetch")
        except Exception:
            print(f"Fetching and embedding {request.repo_url}...")
            repo_data = fetch_repo_data(request.repo_url)
            embed_repo_data(repo_data)

        # Step 2: Retrieve relevant chunks
        results = retrieve_and_rerank(request.question, repo_name)

        # Step 3: Build context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {r['metadata']['source']}]\n{r['text']}"
            for r in results
        ])

        # Step 4: Generate answer with Claude
        prompt = f"""You are an expert OSS onboarding assistant helping beginners contribute to open source projects.
        Use the following context from the repository to answer the question:{context}
        Question: {request.question}
        Give a clear, beginner-friendly answer based only on the context provided."""
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.content[0].text
        sources = [r["metadata"]["source"] for r in results]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}