import os
import re
import base64
import requests
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_headers():
    return {
        "Authorization": f"Bearer {os.getenv('GITHUB_PAT')}",
        "Accept": "application/vnd.github+json"
    }


def fetch_file_content(repo_full_name: str, file_path: str) -> str:
    """Fetch raw file content from GitHub."""
    r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}",
        headers=get_headers()
    )
    if r.status_code != 200:
        return None
    content = r.json()
    if content.get("encoding") == "base64":
        return base64.b64decode(content["content"]).decode("utf-8", errors="replace")
    return None


def chunk_by_functions(code: str, file_path: str) -> list[dict]:
    """
    Chunk Python code by function and class boundaries.
    Each chunk = one complete function or class definition.
    """
    chunks = []
    
    # Match function and class definitions
    pattern = r'((?:^|\n)(?:async\s+)?def\s+\w+|(?:^|\n)class\s+\w+)'
    matches = list(re.finditer(pattern, code))
    
    if not matches:
        # No functions found — treat whole file as one chunk
        chunks.append({
            "text": code[:3000],
            "metadata": {"file": file_path, "type": "module"}
        })
        return chunks
    
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
        chunk_text = code[start:end].strip()
        
        if len(chunk_text) > 50:  # skip trivial chunks
            chunks.append({
                "text": chunk_text[:2000],  # cap at 2000 chars
                "metadata": {
                    "file": file_path,
                    "type": "function" if "def " in match.group() else "class",
                    "name": match.group().strip()
                }
            })
    
    return chunks


def embed_code_files(repo_full_name: str, file_paths: list[str]) -> str:
    """Fetch, chunk, and embed code files into ChromaDB."""
    collection_name = repo_full_name.replace("/", "__") + "__code"
    
    try:
        chroma_client.get_collection(collection_name)
        return collection_name  # already embedded
    except Exception:
        pass
    
    collection = chroma_client.create_collection(collection_name)
    
    all_chunks = []
    for file_path in file_paths:
        content = fetch_file_content(repo_full_name, file_path)
        if content:
            chunks = chunk_by_functions(content, file_path)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        return None
    
    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [r.embedding for r in response.data]
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )
    
    return collection_name


@traceable(name="navigate_codebase")
def navigate_codebase(repo_full_name: str, file_paths: list[str], question: str) -> str:
    """
    Main entry point: embed code files and answer questions about them.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Embed files
    collection_name = embed_code_files(repo_full_name, file_paths)
    if not collection_name:
        return "Could not fetch or embed the specified files."
    
    # Retrieve relevant chunks
    collection = chroma_client.get_collection(collection_name)
    query_response = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"]
    )
    
    # Build context
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        context += f"\n[File: {meta.get('file')} | {meta.get('name', '')}]\n{doc}\n"
    
    # Generate answer
    prompt = f"""You are an expert code reviewer helping a beginner contributor understand a codebase.

Here are the relevant code sections:
{context}

Question: {question}

Explain clearly and point to specific functions, line patterns, or locations where changes should be made.
Be beginner-friendly but technically precise."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text


if __name__ == "__main__":
    result = navigate_codebase(
        repo_full_name="scikit-learn/scikit-learn",
        file_paths=["sklearn/utils/validation.py"],
        question="Where should Ic add input validation for a new parameter?"
    )
    print(result)