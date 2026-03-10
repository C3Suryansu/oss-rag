import os
from github import Auth, Github
from dotenv import load_dotenv

load_dotenv()

def get_github_client():
    token = os.getenv("GITHUB_PAT")
    auth = Auth.Token(token)
    return Github(auth=auth)

def fetch_repo_data(repo_url: str) -> dict:
    """
    Given a GitHub repo URL, fetches:
    - README
    - CONTRIBUTING.md (if exists)
    - Top-level file structure
    - Open issues labeled 'good first issue'
    """
    # Extract owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    owner, repo_name = parts[-2], parts[-1]

    client = get_github_client()
    repo = client.get_repo(f"{owner}/{repo_name}")

    data = {
        "name": repo.full_name,
        "description": repo.description,
        "readme": None,
        "contributing": None,
        "file_structure": [],
        "good_first_issues": []
    }

    # Fetch README
    try:
        readme = repo.get_readme()
        data["readme"] = readme.decoded_content.decode("utf-8")
    except Exception:
        print("No README found")

    #Fetch Contributing.MD
    try:
        contributing = repo.get_contents("CONTRIBUTING.md")
        data["contributing"] = contributing.decoded_content.decode("utf-8")
    except Exception:
        print("No CONTRIBUTING.md found")

    # Fetch top-level file structure
    try:
        contents = repo.get_contents("")
        if isinstance(contents, list):
            data["file_structure"] = [f.path for f in contents]
        else:
            data["file_structure"] = [contents.path]
    except Exception as e:
        print(f"Could not fetch file structure: {e}")
        
    # Fetch good first issues
    try:
        issues = repo.get_issues(state="open", labels=["good first issue"])
        for issue in issues[:20]: # cap at 20
            data["good_first_issues"].append({
                "title": issue.title,
                "url": issue.html_url,
                "body": issue.body[:500] if issue.body else ""
            })
    except Exception:
        print("Could not fetch issues")
    
    return data

if __name__ == "__main__":
    result = fetch_repo_data("https://github.com/kubeflow/pipelines")
    print(f"Repo: {result['name']}")
    print(f"Description: {result['description']}")
    print(f"README length: {len(result['readme'])} chars" if result['readme'] else "No README")
    print(f"Contributing: {'Found' if result['contributing'] else 'Not found'}")
    print(f"Files: {result['file_structure']}")
    print(f"Good first issues: {len(result['good_first_issues'])}")
    for issue in result['good_first_issues'][:3]:
        print(f"  - {issue['title']}")

