import os
import cohere
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from langsmith import traceable

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

@traceable(name="get_query_embedding")
def get_query_embedding(query: str) -> List[float]:
    """Convert user query to embedding vector."""
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

@traceable(name="chromadb_retrieve")
def retrieve(query: str, collection_name: str, top_k: int = 10) -> List[Dict]:
    """
    Stage 1: Semantic search — fetch top_k candidate chunks from ChromaDB.
    """
    collection = chroma_client.get_collection(collection_name)
    query_embedding = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    candidates = []
    for i in range(len(results["documents"][0])):
        candidates.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return candidates

@traceable(name="cohere_rerank")
def rerank(query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
    """
    Stage 2: Cohere reranking — reorder candidates by true relevance.
    """
    documents = [c["text"] for c in candidates]

    response = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n
    )

    reranked = []
    for result in response.results:
        candidate = candidates[result.index]
        candidate["relevance_score"] = result.relevance_score
        reranked.append(candidate)

    return reranked

@traceable(name="retrieve_and_rerank")
def retrieve_and_rerank(query: str, collection_name: str) -> List[Dict]:
    """
    Full two-stage retrieval pipeline.
    """
    candidates = retrieve(query, collection_name, top_k=10)
    reranked = rerank(query, candidates, top_n=3)
    return reranked


if __name__ == "__main__":
    query = "How do I set up kubeflow pipelines locally?"
    collection_name = "kubeflow__pipelines"

    print(f"Query: {query}\n")
    results = retrieve_and_rerank(query, collection_name)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {result['metadata']['source']}")
        print(f"Relevance: {result['relevance_score']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print()