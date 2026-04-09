import streamlit as st
import requests
import os
import re
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="OSS Contribution Assistant",
    page_icon="🚀",
    layout="centered"
)

# ── Multi-conversation state ──────────────────────────────────────────────────
# conversations: list of dicts, each holding one full conversation
# active_idx: which conversation is currently shown

def _new_conversation() -> dict:
    return {
        "id": datetime.now().strftime("%H:%M:%S"),
        "messages": [],
        "phase": "skills",       # setup is done once globally; convos start at skills
        "skills": [],
        "selected_repo": None,
        "selected_issue": None,
    }

def _conv_label(c: dict) -> str:
    if c.get("selected_repo"):
        issue = f" #{c['selected_issue']}" if c.get("selected_issue") else ""
        return f"{c['selected_repo']}{issue}"
    if c.get("skills"):
        return ", ".join(c["skills"][:3])
    return f"New chat ({c['id']})"

# Global keys/passphrase — shared across all conversations in this tab
_key_defaults = {
    "anthropic_key": None,
    "openai_key": None,
    "github_pat": None,
    "admin_passphrase": None,
    "setup_done": False,
}
for k, v in _key_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Conversation list
if "conversations" not in st.session_state:
    st.session_state.conversations = [_new_conversation()]
if "active_idx" not in st.session_state:
    st.session_state.active_idx = 0

# Shorthand to current conversation
def conv() -> dict:
    return st.session_state.conversations[st.session_state.active_idx]

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_message(role: str, content: str):
    conv()["messages"].append({"role": role, "content": content})

def user_headers():
    h = {}
    if st.session_state.admin_passphrase:
        h["X-Admin-Passphrase"] = st.session_state.admin_passphrase
        return h
    if st.session_state.anthropic_key:
        h["X-Anthropic-Key"] = st.session_state.anthropic_key
    if st.session_state.openai_key:
        h["X-OpenAI-Key"] = st.session_state.openai_key
    if st.session_state.github_pat:
        h["X-GitHub-PAT"] = st.session_state.github_pat
    return h

def call_skill_match(skills):
    try:
        r = requests.post(f"{API_URL}/skill-match", json={"skills": skills}, headers=user_headers(), timeout=300)
        return r.json()["result"] if r.status_code == 200 else None
    except Exception:
        return None

def call_analyze_issues(repo, skills):
    try:
        r = requests.post(f"{API_URL}/analyze-issues", json={"repo_full_name": repo, "skills": skills}, headers=user_headers(), timeout=300)
        return r.json()["result"] if r.status_code == 200 else None
    except Exception:
        return None

def call_contribution_agent(repo, issue, skills):
    try:
        r = requests.post(f"{API_URL}/contribution-agent", json={
            "skills": skills, "selected_repo": repo,
            "selected_issue": issue, "question": None
        }, headers=user_headers(), timeout=600)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_query(repo, question):
    try:
        r = requests.post(f"{API_URL}/query", json={
            "repo_url": f"https://github.com/{repo}", "question": question
        }, headers=user_headers(), timeout=120)
        return r.json().get("answer") if r.status_code == 200 else None
    except Exception:
        return None

def extract_repo(text):
    text = text.strip()
    if "github.com/" in text:
        parts = text.split("github.com/")[-1].strip("/").split("/")
        return f"{parts[0]}/{parts[1]}"
    return text

def extract_issue_number(text):
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

# ── Page title ────────────────────────────────────────────────────────────────

st.title("OSS Contribution Assistant")
st.caption("Your AI guide to open source contribution.")

# ── Phase: setup (one-time key collection) ────────────────────────────────────

if not st.session_state.setup_done:
    st.markdown("### Before we start")
    st.markdown(
        "This app uses your own API keys so you're billed directly — "
        "your keys are never stored and only sent to the backend for your session."
    )
    with st.form("setup_form"):
        anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...",
                                      help="console.anthropic.com")
        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-...",
                                   help="platform.openai.com/api-keys")
        github_pat = st.text_input("GitHub Personal Access Token", type="password", placeholder="github_pat_...",
                                   help="github.com/settings/tokens (repo read scope)")
        st.markdown("---")
        admin_passphrase = st.text_input("Admin Passphrase (optional — owner only)", type="password",
                                         placeholder="Leave blank if you're a public user")
        submitted = st.form_submit_button("Start Session", use_container_width=True, type="primary")

    if submitted:
        if admin_passphrase:
            st.session_state.admin_passphrase = admin_passphrase
            st.session_state.setup_done = True
            st.rerun()
        elif not anthropic_key or not openai_key or not github_pat:
            st.error("All three keys are required (or enter the admin passphrase).")
        else:
            st.session_state.anthropic_key = anthropic_key
            st.session_state.openai_key = openai_key
            st.session_state.github_pat = github_pat
            st.session_state.setup_done = True
            st.rerun()

    st.stop()   # don't render chat until keys are entered

# ── Render current conversation ───────────────────────────────────────────────

c = conv()

for msg in c["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Phase: skills ─────────────────────────────────────────────────────────────

if c["phase"] == "skills":
    if not c["messages"]:
        with st.chat_message("assistant"):
            welcome = "Tell me your skills to get started (e.g. Python, machine learning, React)."
            st.markdown(welcome)

    if prompt := st.chat_input("Tell me your skills..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        skills = [s.strip() for s in re.split(r"[,\s]+", prompt) if s.strip()]
        c["skills"] = skills

        with st.chat_message("assistant"):
            with st.spinner("Finding repos that match your skills... (2-3 min)"):
                result = call_skill_match(skills)
            if result:
                response = (
                    f"Here are some repos that match your skills:\n\n{result}\n\n"
                    "**Which repo would you like to explore?** Enter `owner/repo` or paste the GitHub URL."
                )
                st.markdown(response)
                add_message("assistant", response)
                c["phase"] = "repo_select"
            else:
                msg = "Sorry, I couldn't fetch repos right now. Please try again."
                st.error(msg)
                add_message("assistant", msg)
        st.rerun()

# ── Phase: repo_select ───────────────────────────────────────────────────────

elif c["phase"] == "repo_select":
    if prompt := st.chat_input("Enter the repo (e.g. pytorch/pytorch or a GitHub URL)..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        repo = extract_repo(prompt)
        c["selected_repo"] = repo

        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing issues in {repo}... (2-3 min)"):
                result = call_analyze_issues(repo, c["skills"])
            if result:
                response = (
                    f"Here are the best issues for you in **{repo}**:\n\n{result}\n\n"
                    "**Which issue do you want to work on?** Enter the issue number."
                )
                st.markdown(response)
                add_message("assistant", response)
                c["phase"] = "issue_select"
            else:
                msg = f"Couldn't analyze issues for `{repo}`. Check the repo name and try again."
                st.error(msg)
                add_message("assistant", msg)
        st.rerun()

# ── Phase: issue_select ───────────────────────────────────────────────────────

elif c["phase"] == "issue_select":
    if prompt := st.chat_input("Enter the issue number (e.g. 42)..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        issue_number = extract_issue_number(prompt)
        if not issue_number:
            with st.chat_message("assistant"):
                msg = "Please enter a valid issue number (just the number, e.g. `42`)."
                st.markdown(msg)
                add_message("assistant", msg)
        else:
            c["selected_issue"] = issue_number
            with st.chat_message("assistant"):
                with st.spinner(f"Running full analysis on {c['selected_repo']} #{issue_number}... (3-5 min)"):
                    data = call_contribution_agent(c["selected_repo"], issue_number, c["skills"])
                if data:
                    parts = []
                    if data.get("deepdive"):
                        parts.append(f"### Issue Deep Dive\n{data['deepdive']}")
                    if data.get("navigation"):
                        parts.append(f"### Codebase Navigation\n{data['navigation']}")
                    if data.get("file_paths"):
                        files = "\n".join([f"- `{f}`" for f in data["file_paths"]])
                        parts.append(f"### Relevant Files\n{files}")
                    if data.get("contribution_plan"):
                        parts.append(f"### Contribution Plan\n{data['contribution_plan']}")
                    parts.append(
                        "---\n**Have a follow-up question?** Ask me anything about this repo or issue."
                    )
                    response = "\n\n".join(parts)
                    st.markdown(response)
                    add_message("assistant", response)
                    c["phase"] = "done"
                else:
                    msg = "Analysis failed. Please try again."
                    st.error(msg)
                    add_message("assistant", msg)
        st.rerun()

# ── Phase: done ───────────────────────────────────────────────────────────────

elif c["phase"] == "done":
    if prompt := st.chat_input("Ask a follow-up question about this repo or issue..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_query(c["selected_repo"], prompt)
            if answer:
                st.markdown(answer)
                add_message("assistant", answer)
            else:
                msg = "Couldn't get an answer. Try rephrasing your question."
                st.error(msg)
                add_message("assistant", msg)
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Conversations")

    if st.button("+ New Conversation", use_container_width=True, type="primary"):
        st.session_state.conversations.append(_new_conversation())
        st.session_state.active_idx = len(st.session_state.conversations) - 1
        st.rerun()

    st.markdown("---")

    for i, convo in enumerate(st.session_state.conversations):
        label = _conv_label(convo)
        is_active = i == st.session_state.active_idx
        # Highlight active conversation
        if is_active:
            st.markdown(f"**→ {label}**")
        else:
            if st.button(label, key=f"conv_{i}", use_container_width=True):
                st.session_state.active_idx = i
                st.rerun()

    st.markdown("---")
    st.caption("Conversations last for this browser session only.")
