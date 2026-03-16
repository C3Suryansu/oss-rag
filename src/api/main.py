import os
import anthropic
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable

from src.ingestion.github_fetcher import fetch_repo_data
from src.embeddings.embedder import embed_repo_data, chroma_client
from src.retrieval.retriever import retrieve_and_rerank

from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

load_dotenv()

app = FastAPI(title="OSS Onboarding RAG")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
anthropic_async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class QueryRequest(BaseModel):
    repo_url: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@traceable(name="claude_generate")
def generate_answer(context: str, question: str) -> str:
    """Synchronous Claude call used by /query endpoint — fully traceable."""
    prompt = f"""You are an expert OSS onboarding assistant helping beginners contribute to open source projects.
    Use the following context from the repository to answer the question:{context}
    Question: {question}
    Give a clear, beginner-friendly answer based only on the context provided."""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


@traceable(name="oss_rag_query")
def run_rag_pipeline(repo_url: str, question: str) -> dict:
    """Full RAG pipeline — traced as a single top-level span."""
    repo_name = repo_url.rstrip("/").split("/")[-2] + "__" + \
                repo_url.rstrip("/").split("/")[-1]

    try:
        chroma_client.get_collection(repo_name)
    except Exception:
        repo_data = fetch_repo_data(repo_url)
        embed_repo_data(repo_data)

    results = retrieve_and_rerank(question, repo_name)

    context = "\n\n".join([
        f"[Source: {r['metadata']['source']}]\n{r['text']}"
        for r in results
    ])

    answer = generate_answer(context, question)
    sources = [r["metadata"]["source"] for r in results]

    return {"answer": answer, "sources": sources}


@app.post("/query", response_model=QueryResponse)
async def query_repo(request: QueryRequest):
    try:
        result = run_rag_pipeline(request.repo_url, request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_repo_stream(request: QueryRequest):

    async def generate():
        try:
            repo_name = request.repo_url.rstrip("/").split("/")[-2] + "__" + \
                        request.repo_url.rstrip("/").split("/")[-1]

            try:
                chroma_client.get_collection(repo_name)
            except Exception:
                yield "Fetching and embedding repository...\n\n"
                repo_data = fetch_repo_data(request.repo_url)
                embed_repo_data(repo_data)

            yield "Searching relevant context...\n\n"
            results = retrieve_and_rerank(request.question, repo_name)

            context = "\n\n".join([
                f"[Source: {r['metadata']['source']}]\n{r['text']}"
                for r in results
            ])

            prompt = f"""You are an expert OSS onboarding assistant helping beginners contribute to open source projects.
            Use the following context from the repository to answer the question:{context}
            Question: {request.question}
            Give a clear, beginner-friendly answer based only on the context provided."""

            async with anthropic_async_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"Error: {str(e)}"

    return FastAPIStreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}