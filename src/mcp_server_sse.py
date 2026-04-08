"""
MCP server with SSE transport — deployable as a web service.
Exposes the same 4 tools as mcp_server.py but over HTTP instead of stdio.

Run locally:   uvicorn src.mcp_server_sse:starlette_app --port 8502
Deploy:        systemd service on GCP VM, port 8502
"""

import os
import json
import asyncio
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

load_dotenv()

# ── MCP server (same as mcp_server.py) ───────────────────────────────────────

mcp_app = Server("oss-rag-mcp")


@mcp_app.list_tools()
async def list_tools() -> list[types.Tool]:
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
            description="Get full details of a specific GitHub issue including body and comments.",
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
        ),
        types.Tool(
            name="suggest_contribution",
            description=(
                "Use the fine-tuned LoRA/QLoRA Mistral-7B model to generate a detailed "
                "contribution plan for a GitHub issue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_full_name": {"type": "string"},
                    "issue_title": {"type": "string"},
                    "issue_body": {"type": "string"}
                },
                "required": ["repo_full_name", "issue_title", "issue_body"]
            }
        )
    ]


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "fetch_repo_data":
        from src.ingestion.github_fetcher import fetch_repo_data
        data = fetch_repo_data(arguments["repo_url"])
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
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        repo_issue_counts = {}

        for label in ["good first issue", "help wanted"]:
            response = requests.get(
                "https://api.github.com/search/issues",
                headers=headers,
                params={"q": f"{arguments['query']} label:\"{label}\" state:open type:issue", "per_page": 50}
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
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        r = requests.get(
            f"https://api.github.com/repos/{arguments['repo_full_name']}/issues/{arguments['issue_number']}",
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

    elif name == "suggest_contribution":
        try:
            from finetune.inference import get_advisor
            plan = get_advisor().suggest(
                repo=arguments["repo_full_name"],
                issue_title=arguments["issue_title"],
                issue_body=arguments["issue_body"],
            )
            return [types.TextContent(type="text", text=plan)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Fine-tuned model not ready: {e}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── SSE transport (HTTP-based, deployable) ────────────────────────────────────

sse = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_app.run(streams[0], streams[1], mcp_app.create_initialization_options())


starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)
