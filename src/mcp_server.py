import asyncio
import os
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from dotenv import load_dotenv


load_dotenv()

# Initialize MCP server
app = Server("oss-rag-mcp")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Tell Claude what tools are available."""
    return [
        types.Tool(
            name="fetch_repo_data",
            description=(
                "Fetch information about a GitHub repository. "
                "Returns README, CONTRIBUTING.md, file structure, and good-first-issues."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "Full GitHub repository URL (e.g. https://github.com/owner/repo)"
                    }
                },
                "required": ["repo_url"]
            }
        ),
        types.Tool(
            name="search_beginner_issues",
            description=(
                "Search GitHub for repositories with beginner-friendly issues "
                "matching a skill query. Returns ranked repos with issue counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Skill-based search query (e.g. 'pytorch machine learning')"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_issue_details",
            description=(
                "Get full details of a specific GitHub issue including body and comments."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_full_name": {
                        "type": "string",
                        "description": "Repository in owner/repo format (e.g. pytorch/pytorch)"
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number"
                    }
                },
                "required": ["repo_full_name", "issue_number"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool call from Claude."""

    if name == "fetch_repo_data":
        from src.ingestion.github_fetcher import fetch_repo_data
        repo_url = arguments["repo_url"]
        data = fetch_repo_data(repo_url)
        result = {
            "name": data.get("name"),
            "description": data.get("description"),
            "readme_preview": (data.get("readme") or "")[:500],
            "file_structure": data.get("file_structure", [])[:20],
            "good_first_issues": data.get("good_first_issues", [])[:5]
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "search_beginner_issues":
        import requests
        token = os.getenv("GITHUB_PAT")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
        query = arguments["query"]
        repo_issue_counts = {}

        for label in ["good first issue", "help wanted"]:
            response = requests.get(
                "https://api.github.com/search/issues",
                headers=headers,
                params={
                    "q": f"{query} label:\"{label}\" state:open type:issue",
                    "per_page": 50
                }
            )
            if response.status_code == 200:
                for issue in response.json().get("items", []):
                    repo_name = issue["repository_url"].split("/repos/")[1]
                    repo_issue_counts[repo_name] = repo_issue_counts.get(repo_name, 0) + 1

        sorted_repos = sorted(repo_issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        result = [{"repo": r, "beginner_issues": c} for r, c in sorted_repos]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "get_issue_details":
        import requests
        token = os.getenv("GITHUB_PAT")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
        repo = arguments["repo_full_name"]
        issue_num = arguments["issue_number"]

        r = requests.get(
            f"https://api.github.com/repos/{repo}/issues/{issue_num}",
            headers=headers
        )
        if r.status_code != 200:
            return [types.TextContent(type="text", text=f"Error: {r.status_code}")]

        issue = r.json()
        result = {
            "title": issue["title"],
            "body": (issue.get("body") or "")[:1000],
            "labels": [l["name"] for l in issue.get("labels", [])],
            "comments": issue.get("comments", 0),
            "url": issue["html_url"]
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())