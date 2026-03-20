import os
import requests
from datetime import datetime, timezone
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- Tools ---

@tool("fetch_repo_issues")
def fetch_repo_issues(repo_full_name: str) -> str:
    """Fetch open issues from a GitHub repository with full details.
    Returns issue title, body, labels, comment count, age, and URL."""
    token = os.getenv("GITHUB_PAT")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/issues",
        headers=headers,
        params={"state": "open", "per_page": 30, "sort": "created", "direction": "desc"}
    )

    if response.status_code != 200:
        return f"Error fetching issues: {response.status_code}"

    issues = response.json()
    # Filter out pull requests (GitHub returns PRs in issues endpoint)
    issues = [i for i in issues if "pull_request" not in i]

    result = []
    now = datetime.now(timezone.utc)

    for issue in issues[:20]:
        created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        age_days = (now - created).days
        labels = [l["name"] for l in issue.get("labels", [])]
        has_assignee = issue.get("assignee") is not None

        result.append(
            f"Issue #{issue['number']}: {issue['title']}\n"
            f"  Labels: {', '.join(labels) if labels else 'none'}\n"
            f"  Comments: {issue['comments']}\n"
            f"  Age: {age_days} days\n"
            f"  Assigned: {has_assignee}\n"
            f"  Body: {issue['body'][:300] if issue['body'] else 'No description'}\n"
            f"  URL: {issue['html_url']}\n"
        )

    return "\n".join(result) if result else "No open issues found."


@tool("score_issue")
def score_issue(issue_number: int, repo_full_name: str) -> str:
    """Fetch full details of a specific issue and score it for beginner suitability."""
    token = os.getenv("GITHUB_PAT")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # Fetch issue details
    r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}",
        headers=headers
    )
    if r.status_code != 200:
        return f"Error fetching issue #{issue_number}"

    issue = r.json()

    # Fetch comments
    comments_r = requests.get(
        f"https://api.github.com/repos/{repo_full_name}/issues/{issue_number}/comments",
        headers=headers
    )
    comments = comments_r.json() if comments_r.status_code == 200 else []

    labels = [l["name"] for l in issue.get("labels", [])]
    body_length = len(issue.get("body") or "")
    comment_count = issue.get("comments", 0)
    has_assignee = issue.get("assignee") is not None

    now = datetime.now(timezone.utc)
    created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
    age_days = (now - created).days

    # Scoring
    score = 50  # base score

    # Label bonuses
    beginner_labels = ["good first issue", "good-first-issue", "beginner", "easy", "starter"]
    if any(l.lower() in beginner_labels for l in labels):
        score += 30

    # Penalize complexity signals
    if body_length > 1000:
        score -= 10
    if comment_count > 10:
        score -= 15
    if age_days > 365:
        score -= 10

    # Penalize if already assigned
    if has_assignee:
        score -= 40

    # Bonus if recent
    if age_days < 30:
        score += 10

    score = max(0, min(score, 100))

    return (
        f"Issue #{issue_number}: {issue['title']}\n"
        f"Labels: {', '.join(labels) if labels else 'none'}\n"
        f"Body length: {body_length} chars\n"
        f"Comments: {comment_count}\n"
        f"Age: {age_days} days\n"
        f"Assigned: {has_assignee}\n"
        f"Beginner Score: {score}/100\n"
        f"URL: {issue['html_url']}"
    )


# --- Agent ---

issue_analyst = Agent(
    role="Issue Analyst",
    goal="Analyze GitHub issues and recommend the top 2-3 most suitable ones for a beginner contributor",
    backstory=(
        "You are an expert open source mentor who helps beginners find the right issues to work on. "
        "You evaluate issues based on difficulty signals, beginner-friendliness, and whether they're "
        "already being worked on. You always recommend issues that are genuinely approachable."
    ),
    tools=[fetch_repo_issues, score_issue],
    verbose=True,
    llm="anthropic/claude-sonnet-4-6"
)


# --- Task + Crew ---

def analyze_issues(repo_full_name: str, skills: list[str]) -> str:
    """Main entry point: takes a repo and user skills, returns top 2-3 recommended issues."""
    skills_str = ", ".join(skills)

    task = Task(
        description=(
            f"Analyze open issues in the GitHub repository '{repo_full_name}' "
            f"for a contributor with these skills: {skills_str}.\n\n"
            f"Steps:\n"
            f"1. Use fetch_repo_issues to get all open issues\n"
            f"2. Use score_issue on the most promising ones (at least 10. Try to atleast get 4-5 repos with score greater than 50 )\n"
            f"3. Return the top 5 issues ranked by beginner score, try to atleast include 2 repos with greater than 50 score. \n\n"
            f"For each recommended issue include: title, score, why it's suitable, and the URL."
        ),
        expected_output=(
            "Top 2-3 recommended issues with: issue title, beginner score, "
            "reason it's suitable for the contributor, and direct URL."
        ),
        agent=issue_analyst
    )

    crew = Crew(
        agents=[issue_analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    repo = "kubeflow/kubeflow"
    skills = ["python", "machine learning", "pytorch"]
    print(analyze_issues(repo, skills))