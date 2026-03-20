import os
import re
import requests
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()


def get_headers():
    return {
        "Authorization": f"Bearer {os.getenv('GITHUB_PAT')}",
        "Accept": "application/vnd.github+json"
    }


@tool("fetch_full_issue")
def fetch_full_issue(repo_full_name: str, issue_number: int) -> str:
    """Fetch complete details of a GitHub issue including body and all comments."""
    headers = get_headers()

    # Fetch issue
    r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}",
        headers=headers
    )
    if r.status_code != 200:
        return f"Error fetching issue: {r.status_code}"

    issue = r.json()

    # Fetch comments
    comments_r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments",
        headers=headers
    )
    comments = comments_r.json() if comments_r.status_code == 200 else []

    result = f"Title: {issue['title']}\n"
    result += f"State: {issue['state']}\n"
    result += f"Labels: {', '.join([l['name'] for l in issue.get('labels', [])])}\n"
    result += f"URL: {issue['html_url']}\n\n"
    result += f"Description:\n{issue.get('body', 'No description')}\n\n"

    if comments:
        result += f"Comments ({len(comments)}):\n"
        for i, comment in enumerate(comments[:10]):
            result += f"\n[{comment['user']['login']}]: {comment['body'][:500]}\n"

    return result


@tool("extract_file_references")
def extract_file_references(text: str) -> str:
    """Extract file paths, function names, and code references from issue text."""
    # Match file paths like src/foo/bar.py or path/to/file.js
    file_pattern = r'[\w\-/]+\.\w{1,6}'
    files = re.findall(file_pattern, text)

    # Match function/method names like `function_name()` or `ClassName.method`
    func_pattern = r'`([^`]+)`'
    code_refs = re.findall(func_pattern, text)

    # Match stack trace lines
    stack_pattern = r'File "([^"]+)", line (\d+)'
    stack_refs = re.findall(stack_pattern, text)

    result = []
    if files:
        result.append(f"File references found: {', '.join(set(files[:20]))}")
    if code_refs:
        result.append(f"Code references: {', '.join(set(code_refs[:20]))}")
    if stack_refs:
        result.append(f"Stack trace locations: {stack_refs}")
    if not result:
        result.append("No explicit file references found in the text.")

    return "\n".join(result)


@tool("fetch_file_content")
def fetch_file_content(repo_full_name: str, file_path: str) -> str:
    """Fetch the content of a specific file from a GitHub repository."""
    headers = get_headers()

    r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}",
        headers=headers
    )
    if r.status_code != 200:
        return f"Could not fetch {file_path}: {r.status_code}"

    import base64
    content = r.json()
    if content.get("encoding") == "base64":
        decoded = base64.b64decode(content["content"]).decode("utf-8", errors="replace")
        return f"File: {file_path}\n\n{decoded[:3000]}"

    return f"Could not decode {file_path}"


# --- Agent ---

deepdive_agent = Agent(
    role="Issue Deep-Dive Specialist",
    goal=(
        "Given a GitHub issue, produce a complete guide for a beginner contributor: "
        "what the issue is about, how to reproduce it, which files to look at, and where the fix likely lives."
    ),
    backstory=(
        "You are an experienced open source mentor. You read GitHub issues thoroughly, "
        "extract all technical clues, and translate them into clear, actionable guidance "
        "for beginners. You always point to specific files and functions."
    ),
    tools=[fetch_full_issue, extract_file_references, fetch_file_content],
    verbose=True,
    llm="anthropic/claude-sonnet-4-6"
)


def deepdive_issue(repo_full_name: str, issue_number: int) -> str:
    """Main entry point: deep dive on a specific issue."""

    task = Task(
        description=(
            f"Do a complete deep-dive on issue #{issue_number} in '{repo_full_name}'.\n\n"
            f"Steps:\n"
            f"1. Use fetch_full_issue to read the complete issue and comments\n"
            f"2. Use extract_file_references to find any file/code references in the issue text\n"
            f"3. If specific files are mentioned, use fetch_file_content to read the relevant ones\n"
            f"4. Produce a structured guide with:\n"
            f"   - What the issue is about (plain English)\n"
            f"   - Steps to reproduce (if applicable)\n"
            f"   - Which files/functions are relevant\n"
            f"   - Where the fix likely needs to go\n"
            f"   - Suggested first step for a beginner contributor"
        ),
        expected_output=(
            "A structured beginner guide with: issue summary, reproduction steps, "
            "relevant files/functions, where the fix goes, and suggested first step."
        ),
        agent=deepdive_agent
    )

    crew = Crew(
        agents=[deepdive_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    # Test with a known scikit-learn issue
    print(deepdive_issue("scikit-learn/scikit-learn", 33582))