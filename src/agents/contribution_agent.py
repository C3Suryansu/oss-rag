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
    contribution_plan: str        # output from fine-tuned advisor (LoRA/QLoRA Mistral-7B)
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
        return {**state, "navigation": "No specific files identified.", "next_step": "finetuned_advisor"}

    question = state.get("question") or "Where should I make changes to fix this issue?"
    result = navigate_codebase(state["selected_repo"], state["file_paths"], question)
    return {**state, "navigation": result, "next_step": "finetuned_advisor"}


def finetuned_advisor_node(state: AgentState) -> AgentState:
    """
    Fine-tuned LoRA/QLoRA Mistral-7B advisor node.
    Uses the deepdive + navigation context to generate a contribution plan.
    Falls back gracefully if MLX adapter isn't trained yet.
    """
    print(f"\n🤖 Fine-tuned advisor generating contribution plan...")
    try:
        from finetune.inference import get_advisor
        advisor = get_advisor()
        # Use deepdive output as issue body — it contains the full issue context
        plan = advisor.suggest(
            repo=state.get("selected_repo", ""),
            issue_title=f"Issue #{state.get('selected_issue', '')}",
            issue_body=(state.get("deepdive", "") + "\n\n" + state.get("navigation", ""))[:1500],
        )
        print("  ✓ Fine-tuned plan generated")
    except Exception as e:
        print(f"  ⚠️  Fine-tuned model not ready ({e}) — skipping, adapter not trained yet")
        plan = ""
    return {**state, "contribution_plan": plan, "next_step": "end"}


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
    graph.add_node("finetuned_advisor", finetuned_advisor_node)

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
    graph.add_conditional_edges("navigate", router, {
        "finetuned_advisor": "finetuned_advisor",
        "end": END
    })
    graph.add_edge("finetuned_advisor", END)


    # MemorySaver enables interrupt_before to work
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["issue_analyze", "deepdive"]
    )


def run_contribution_agent(
    skills: list[str],
    selected_repo: str,
    selected_issue: int,
    question: str = None
) -> dict:
    """Run the full pipeline non-interactively (used by the API endpoint)."""
    # Build a graph without interrupt_before so it executes end-to-end in one call
    g = StateGraph(AgentState)
    g.add_node("issue_analyze", issue_analyze_node)
    g.add_node("deepdive", deepdive_node)
    g.add_node("navigate", navigate_node)
    g.add_node("finetuned_advisor", finetuned_advisor_node)
    g.set_entry_point("issue_analyze")
    g.add_conditional_edges("issue_analyze", router, {"select_issue": "deepdive", "end": END})
    g.add_conditional_edges("deepdive", router, {"navigate": "navigate", "end": END})
    g.add_conditional_edges("navigate", router, {"finetuned_advisor": "finetuned_advisor", "end": END})
    g.add_edge("finetuned_advisor", END)
    compiled = g.compile()

    initial_state = AgentState(
        skills=skills,
        repos="",
        selected_repo=selected_repo,
        issues="",
        selected_issue=selected_issue,
        deepdive="",
        file_paths=[],
        navigation="",
        contribution_plan="",
        question=question or "Where should I make changes to fix this issue?",
        messages=[],
        next_step="select_issue"
    )

    return compiled.invoke(initial_state)

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
    print("\n✅ Contribution guide complete!\n")
    print("Ask follow-up questions about the code, or type 'done' to exit.\n")

    while True:
        follow_up = input("👉 Your question: ").strip()
        if follow_up.lower() in ["done", "exit", "quit", ""]:
            print("\n🎉 Good luck with your contribution!")
            break

        from src.agents.codebase_navigator import navigate_codebase
        if result.get("file_paths"):
            answer = navigate_codebase(
                result["selected_repo"],
                result["file_paths"],
                follow_up
            )
        else:
            # No files extracted — use Claude directly
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Context: We are working on {result['selected_repo']} issue #{result['selected_issue']}.\n\nDeep dive summary:\n{result['deepdive'][:500]}\n\nQuestion: {follow_up}"}]
            )
            answer = response.content[0].text

        print(f"\n{answer}\n")
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
