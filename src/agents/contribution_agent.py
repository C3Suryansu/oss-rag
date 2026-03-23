import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# --- State Schema ---

class AgentState(TypedDict):
    skills: list[str]
    repos: str                    # output from skill matcher
    selected_repo: str            # user picks one
    issues: str                   # output from issue analyzer
    selected_issue: int           # user picks one
    deepdive: str                 # output from deep dive
    file_paths: list[str]         # extracted from deep dive
    navigation: str               # output from codebase navigator
    question: str                 # user's code question
    messages: Annotated[list, add_messages]  # conversation history
    next_step: str                # routing signal


# --- Nodes ---

def skill_match_node(state: AgentState) -> AgentState:
    """Find repos matching user's skills."""
    from src.agents.skill_matcher import match_skills_to_repos
    print(f"\n🔍 Finding repos for skills: {state['skills']}")
    result = match_skills_to_repos(state["skills"])
    return {**state, "repos": result, "next_step": "select_repo"}


def issue_analyze_node(state: AgentState) -> AgentState:
    """Analyze issues for selected repo."""
    from src.agents.issue_analyzer import analyze_issues
    print(f"\n🐛 Analyzing issues for: {state['selected_repo']}")
    result = analyze_issues(state["selected_repo"], state["skills"])
    return {**state, "issues": result, "next_step": "select_issue"}


def deepdive_node(state: AgentState) -> AgentState:
    """Deep dive on selected issue."""
    from src.agents.issue_deepdive import deepdive_issue
    print(f"\n🔬 Deep diving on issue #{state['selected_issue']}")
    result = deepdive_issue(state["selected_repo"], state["selected_issue"])
    
    # Extract file paths from deep dive output (simple heuristic)
    import re
    files = re.findall(r'[\w\-/]+\.(?:py|js|ts|go|java|rs|cpp|c|rb)\b', result)
    file_paths = list(set(files[:5]))  # top 5 unique files
    
    return {**state, "deepdive": result, "file_paths": file_paths, "next_step": "navigate"}


def navigate_node(state: AgentState) -> AgentState:
    """Navigate codebase for relevant files."""
    from src.agents.codebase_navigator import navigate_codebase
    print(f"\n🧭 Navigating codebase: {state['file_paths']}")
    
    if not state.get("file_paths"):
        return {**state, "navigation": "No specific files identified.", "next_step": "end"}
    
    question = state.get("question") or "Where should I make changes to fix this issue?"
    result = navigate_codebase(state["selected_repo"], state["file_paths"], question)
    return {**state, "navigation": result, "next_step": "end"}


# --- Routing ---

def router(state: AgentState) -> str:
    return state.get("next_step", "end")


# --- Build Graph ---

def build_graph(memory = None):
    graph = StateGraph(AgentState)

    graph.add_node("skill_match", skill_match_node)
    graph.add_node("issue_analyze", issue_analyze_node)
    graph.add_node("deepdive", deepdive_node)
    graph.add_node("navigate", navigate_node)

    graph.set_entry_point("skill_match")

    graph.add_conditional_edges("skill_match", router, {
        "select_repo": "issue_analyze",
        "end": END
    })
    graph.add_conditional_edges("issue_analyze", router, {
        "select_issue": "deepdive",
        "end": END
    })
    graph.add_conditional_edges("deepdive", router, {
        "navigate": "navigate",
        "end": END
    })
    graph.add_edge("navigate", END)


    # MemorySaver enables interrupt_before to work
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["issue_analyze", "deepdive"]
    )


""" Commenting this part -> Used for streamlit segment on the UI
def run_contribution_agent(
    skills: list[str],
    selected_repo: str,
    selected_issue: int,
    question: str = None
) -> dict:
    # Run the full contribution agent pipeline.
    graph = build_graph()

    initial_state = AgentState(
        skills=skills,
        repos="",
        selected_repo=selected_repo,
        issues="",
        selected_issue=selected_issue,
        deepdive="",
        file_paths=[],
        navigation="",
        question=question or "Where should I make changes to fix this issue?",
        messages=[],
        next_step="select_repo"
    )

    # Start from issue_analyze since we already have repo + issue
    result = graph.invoke(initial_state)
    return result
"""

# For the conversational chatbot
def run_conversational_agent(skills: list[str]):
    """Run the contribution agent conversationally with human-in-the-loop."""
    memory = MemorySaver()
    graph = build_graph(memory=memory)
    thread_id = "session_1"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentState(
        skills=skills,
        repos="",
        selected_repo="",
        issues="",
        selected_issue=0,
        deepdive="",
        file_paths=[],
        navigation="",
        question="Where should I make changes to fix this issue?",
        messages=[],
        next_step="select_repo"
    )

    print("\n🚀 Starting OSS Contribution Agent...\n")

    # Step 1: Run skill matching — pauses before issue_analyze
    graph.invoke(initial_state, config=config)
    state = graph.get_state(config).values

    print("\n📦 Repos found:")
    print(state["repos"])

    # Step 2: User picks a repo
    # Step 2: User picks a repo
    selected_repo = input("\n👉 Enter the repo (owner/repo format): ").strip()

    # Only update selected_repo — let issue_analyze_node set next_step naturally
    graph.update_state(config, {"selected_repo": selected_repo})

    # Resume — runs issue_analyze, pauses before deepdive
    graph.invoke(None, config=config)
    state = graph.get_state(config).values

    print("\n🐛 Issues found:")
    print(state["issues"])

    # Step 3: User picks an issue
    issue_number = int(input("\n👉 Enter the issue number: ").strip())

    # Only update selected_issue — let deepdive_node set next_step naturally
    graph.update_state(config, {"selected_issue": issue_number})

    # Resume — runs deepdive + navigate
    result = graph.invoke(None, config=config)

    print("\n=== DEEP DIVE ===")
    print(result["deepdive"][:1000])
    print("\n=== CODE NAVIGATION ===")
    print(result["navigation"][:1000])

    return result

if __name__ == "__main__":
    # Visualize the graph
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
    print("\n---\n")
    # For the conversational agent

    run_conversational_agent(skills=["python", "machine learning", "pytorch"])

    """ For the UI part
    # Test run
    result = run_contribution_agent(
        skills=["python", "machine learning"],
        selected_repo="scikit-learn/scikit-learn",
        selected_issue=29243,
        question="Where should I add the fix for this issue?"
    )
    print("\n=== DEEP DIVE ===")
    print(result["deepdive"][:500])
    print("\n=== NAVIGATION ===")
    print(result["navigation"][:500])
    """
