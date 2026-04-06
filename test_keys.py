# Added to create testing keys 
import os
from dotenv import load_dotenv

load_dotenv()

def test_anthropic():
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role":"user", "content":"say hi"}]
    )
    print("Anthropic:", response.content[0].text)

def test_openai():
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-small"
    )
    print("OpenAI embeddings: vector length =", len(response.data[0].embedding))

def test_cohere():
    import cohere
    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    response = client.rerank(
        model="rerank-v3.5",
        query="how to contribute",
        documents=["fork the repo", "buy a coffee", "submit a pull request"],
        top_n=2            
    )
    print("Cohere rerank: top result =", response.results[0].index)

def test_github():
    from github import Github
    client=Github(os.getenv("GITHUB_PAT"))
    repo=client.get_repo("tiangolo/fastapi")
    print("Github PAT: repo", repo.full_name)

if __name__ == "__main__":
    test_anthropic()
    test_openai()
    test_cohere()
    test_github()
