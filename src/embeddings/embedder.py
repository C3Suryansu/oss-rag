import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

chroma_client = chromadb.PersistentClient(path = "./chroma_db")

def _openai():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Docstring for chunk_text
    
    :param text: Description
    :type text: str
    :param chunk_size: Description
    :type chunk_size: int
    :param overlap: Description
    :type overlap: int
    :return: Description
    :rtype: List[str]

    Splits text into overlapping chunks
    Chunk size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> List[float]:
    """Generate OpenAI embedding for a single text."""
    response = _openai().embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def get_embedding(text: str) -> List[float]:
    """Generate OpenAI embedding for a single text."""
    response = _openai().embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def embed_repo_data(repo_data: dict) -> str:
    """
    Takes repo data from github_fetcher, chunks it,
    embeds each chunk, and stores in ChromaDB.
    Returns the collection name.
    """
    repo_name = repo_data["name"].replace("/", "__")
    collection = chroma_client.get_or_create_collection(name=repo_name)

    documents = []
    metadatas = []
    ids = []
    chunk_id = 0

    # Chunk and embed README
    if repo_data.get("readme"):
        for chunk in chunk_text(repo_data["readme"]):
            documents.append(chunk)
            metadatas.append({"source": "readme", "repo": repo_data["name"]})
            ids.append(f"readme_{chunk_id}")
            chunk_id += 1

    # Chunk and embed CONTRIBUTING
    if repo_data.get("contributing"):
        for chunk in chunk_text(repo_data["contributing"]):
            documents.append(chunk)
            metadatas.append({"source": "contributing", "repo": repo_data["name"]})
            ids.append(f"contributing_{chunk_id}")
            chunk_id += 1

    # Embed each good first issue as its own document
    for i, issue in enumerate(repo_data.get("good_first_issues", [])):
        issue_text = f"Issue: {issue['title']}\n{issue['body']}"
        documents.append(issue_text)
        metadatas.append({"source": "issue", "repo": repo_data["name"], "url": issue["url"]})
        ids.append(f"issue_{i}")

    # Generate embeddings in batch
    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = [get_embedding(doc) for doc in documents]

    # Store in ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Stored {len(documents)} chunks in ChromaDB collection: {repo_name}")
    return repo_name

if __name__ == "__main__":
    from src.ingestion.github_fetcher import fetch_repo_data

    print("Fetching repo data...")
    repo_data = fetch_repo_data("https://github.com/kubeflow/pipelines")

    print("Embedding and storing...")
    collection_name = embed_repo_data(repo_data)

    print(f"\nDone. Collection: {collection_name}")

    # Verify it's stored
    collection = chroma_client.get_collection(collection_name)
    print(f"Total chunks stored: {collection.count()}")
