import os
from github import Auth, Github
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

# --- Tools ---

@tool("search_github_repos")
def search_github_repos(query: str) -> str:
    """Search GitHub for repositories with beginner-friendly issues matching the given skills."""
    token = os.getenv("GITHUB_PAT")
    auth = Auth.Token(token)
    client = Github(auth=auth)

    BEGINNER_LABELS = [
        "good first issue",
        "good-first-issue",
        "beginner friendly",
        "help wanted",
        "first-timers-only",
    ]

    repo_issue_counts = {}  # repo_full_name -> issue count

    for label in BEGINNER_LABELS:
        try:
            issues = client.search_issues(
                query=f"{query} label:\"{label}\" state:open",
                sort="created"
            )
            count = 0
            for issue in issues:
                if count >= 50:
                    break
                repo_name = issue.repository.full_name
                repo_issue_counts[repo_name] = repo_issue_counts.get(repo_name, 0) + 1
                count += 1
        except Exception:
            continue

    # Sort repos by number of beginner issues found
    sorted_repos = sorted(repo_issue_counts.items(), key=lambda x: x[1], reverse=True)

    results = []
    for repo_name, count in sorted_repos[:10]:
        try:
            repo = client.get_repo(repo_name)
            results.append(
                f"- {repo_name} ({repo.stargazers_count} stars, {count} beginner issues found)\n"
                f"  {repo.description}\n"
                f"  URL: {repo.html_url}"
            )
        except Exception:
            continue

    return "\n".join(results) if results else "No repos found."

@tool("score_repo_for_contributors")
def score_repo_for_contributors(repo_full_name: str) -> str:
    """Score a GitHub repo by its contributor-friendliness."""
    token = os.getenv("GITHUB_PAT")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    BEGINNER_LABELS = [
        "good first issue", "good-first-issue",
        "beginner friendly", "beginner", "easy",
        "help wanted", "first-timers-only", "up-for-grabs"
    ]

    try:
        r = requests.get(f"https://api.github.com/repos/{repo_full_name}", headers=headers)
        repo = r.json()

        # Count beginner issues across all labels
        beginner_issues = set()
        for label in BEGINNER_LABELS:
            resp = requests.get(
                f"https://api.github.com/repos/{repo_full_name}/issues",
                headers=headers,
                params={"state": "open", "labels": label, "per_page": 50}
            )
            if resp.status_code == 200:
                for issue in resp.json():
                    beginner_issues.add(issue["number"])

        total_beginner = len(beginner_issues)

        # Check CONTRIBUTING.md
        contrib_r = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/contents/CONTRIBUTING.md",
            headers=headers
        )
        has_contributing = contrib_r.status_code == 200

        last_push = repo.get("pushed_at", "Unknown")[:10]
        stars = repo.get("stargazers_count", 0)

        score = 0
        score += min(total_beginner * 10, 40)
        score += 30 if has_contributing else 0
        score += 30 if stars > 500 else 15

        return (
            f"Repo: {repo_full_name}\n"
            f"Beginner issues (all labels): {total_beginner}\n"
            f"Has CONTRIBUTING.md: {has_contributing}\n"
            f"Last push: {last_push}\n"
            f"Stars: {stars}\n"
            f"Contributor Score: {score}/100"
        )
    except Exception as e:
        return f"Error scoring {repo_full_name}: {str(e)}"


# --- Agents ---

researcher = Agent(
    role="GitHub Researcher",
    goal="Find open source repositories on GitHub that match the user's technical skills",
    backstory=(
        "You are an expert at discovering relevant open source projects with great first issues count. "
        "You search GitHub strategically using skill-based queries and "
        "identify repos that are active, well-maintained, beginner-friendly and has good first issues open."
    ),
    tools=[search_github_repos],
    verbose=True,
    llm="anthropic/claude-sonnet-4-6"
)

scorer = Agent(
    role="Repo Scorer",
    goal="Score and rank repositories by how contributor-friendly they are",
    backstory=(
        "You evaluate open source repositories from the perspective of a new contributor. "
        "You check for good-first-issues, CONTRIBUTING.md, recent activity, and community size. "
        "You produce a ranked list with clear reasoning."
    ),
    tools=[score_repo_for_contributors],
    verbose=True,
    llm="anthropic/claude-sonnet-4-6"
)


# --- Tasks ---

def build_crew(skills: list[str]) -> Crew:
    skills_str = ", ".join(skills)

    research_task = Task(
        description=(
            f"Search GitHub for open source repositories matching these skills: {skills_str}. "
            f"Find at least 5 relevant, active repositories. "
            f"Return the full_name (owner/repo format) and a brief description of each."
        ),
        expected_output="A list of 5+ GitHub repos with their full_name and description.",
        agent=researcher
    )

    scoring_task = Task(
        description=(
            "Score each repository found by the researcher for contributor-friendliness. "
            "Use the score_repo_for_contributors tool on each repo. "
            "Return the top 3 repos ranked by their contributor score with reasoning."
        ),
        expected_output=(
            "Top 3 repos ranked by contributor score. "
            "For each: repo name, score, good-first-issues count, and why it's recommended."
        ),
        agent=scorer,
        context=[research_task]
    )

    return Crew(
        agents=[researcher, scorer],
        tasks=[research_task, scoring_task],
        process=Process.sequential,
        verbose=True
    )


def match_skills_to_repos(skills: list[str]) -> str:
    """Main entry point: takes a list of skills, returns ranked repos."""
    crew = build_crew(skills)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    skills = ["python", "machine learning", "pytorch"]
    print(match_skills_to_repos(skills))