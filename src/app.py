import streamlit as st
import requests
import os
import re

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="OSS Contribution Assistant",
    page_icon="🚀",
    layout="centered"
)

st.title("OSS Contribution Assistant")
st.caption("Your AI guide to open source contribution. Tell me your skills to get started.")

# ── Session state init ─────────────────────────────────────────────────────────

defaults = {
    "messages": [],
    "phase": "setup",        # setup → skills → repo_select → issue_select → done
    "skills": [],
    "selected_repo": None,
    "selected_issue": None,
    "anthropic_key": None,
    "openai_key": None,
    "github_pat": None,
    "admin_passphrase": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def user_headers():
    """Build request headers from session. Sends admin passphrase if set, otherwise user API keys."""
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
            "skills": skills,
            "selected_repo": repo,
            "selected_issue": issue,
            "question": None
        }, headers=user_headers(), timeout=600)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_query(repo, question):
    try:
        r = requests.post(f"{API_URL}/query", json={
            "repo_url": f"https://github.com/{repo}",
            "question": question
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

# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Phase: setup (key collection) ────────────────────────────────────────────

if st.session_state.phase == "setup":
    st.markdown("### Before we start")
    st.markdown(
        "This app uses your own API keys so you're billed directly — "
        "your keys are never stored and only sent to the backend for your session."
    )

    with st.form("setup_form"):
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get yours at console.anthropic.com"
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="Get yours at platform.openai.com/api-keys"
        )
        github_pat = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            placeholder="github_pat_...",
            help="Get yours at github.com/settings/tokens (repo read scope)"
        )
        st.markdown("---")
        admin_passphrase = st.text_input(
            "Admin Passphrase (optional — owner only)",
            type="password",
            placeholder="Leave blank if you're a public user",
            help="If you have the admin passphrase, enter it here instead of the keys above."
        )
        submitted = st.form_submit_button("Start Session", use_container_width=True, type="primary")

    if submitted:
        if admin_passphrase:
            # Admin path — passphrase bypasses key requirement
            st.session_state.admin_passphrase = admin_passphrase
            st.session_state.phase = "skills"
            welcome = "Admin session started. **Tell me your skills to get started** (e.g. Python, machine learning, React)."
            add_message("assistant", welcome)
            st.rerun()
        elif not anthropic_key or not openai_key or not github_pat:
            st.error("All three keys are required (or enter the admin passphrase).")
        else:
            st.session_state.anthropic_key = anthropic_key
            st.session_state.openai_key = openai_key
            st.session_state.github_pat = github_pat
            st.session_state.phase = "skills"
            welcome = "Keys saved for this session. **Tell me your skills to get started** (e.g. Python, machine learning, React)."
            add_message("assistant", welcome)
            st.rerun()

# ── Phase: skills ─────────────────────────────────────────────────────────────

elif st.session_state.phase == "skills":
    if prompt := st.chat_input("Tell me your skills (e.g. Python, machine learning, React)..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        skills = [s.strip() for s in re.split(r"[,\s]+", prompt) if s.strip()]
        st.session_state.skills = skills

        with st.chat_message("assistant"):
            with st.spinner("Finding repos that match your skills... (2-3 min)"):
                result = call_skill_match(skills)

            if result:
                response = (
                    f"Here are some repos that match your skills:\n\n{result}\n\n"
                    "**Which repo would you like to explore?** "
                    "Enter the name in `owner/repo` format or paste the GitHub URL."
                )
                st.markdown(response)
                add_message("assistant", response)
                st.session_state.phase = "repo_select"
            else:
                msg = "Sorry, I couldn't fetch repos right now. Please try again."
                st.error(msg)
                add_message("assistant", msg)

        st.rerun()

# ── Phase: repo_select ───────────────────────────────────────────────────────

elif st.session_state.phase == "repo_select":
    if prompt := st.chat_input("Enter the repo (e.g. pytorch/pytorch or a GitHub URL)..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        repo = extract_repo(prompt)
        st.session_state.selected_repo = repo

        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing issues in {repo}... (2-3 min)"):
                result = call_analyze_issues(repo, st.session_state.skills)

            if result:
                response = (
                    f"Here are the best issues for you in **{repo}**:\n\n{result}\n\n"
                    "**Which issue do you want to work on?** Enter the issue number."
                )
                st.markdown(response)
                add_message("assistant", response)
                st.session_state.phase = "issue_select"
            else:
                msg = f"Couldn't analyze issues for `{repo}`. Check the repo name and try again."
                st.error(msg)
                add_message("assistant", msg)

        st.rerun()

# ── Phase: issue_select ───────────────────────────────────────────────────────

elif st.session_state.phase == "issue_select":
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
            st.session_state.selected_issue = issue_number
            repo = st.session_state.selected_repo

            with st.chat_message("assistant"):
                with st.spinner(f"Running full analysis on {repo} #{issue_number}... (3-5 min)"):
                    data = call_contribution_agent(repo, issue_number, st.session_state.skills)

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
                        "---\n**Have a follow-up question?** Ask me anything about this repo or issue. "
                        "Or use the sidebar to start a new conversation."
                    )

                    response = "\n\n".join(parts)
                    st.markdown(response)
                    add_message("assistant", response)
                    st.session_state.phase = "done"
                else:
                    msg = "Analysis failed. Please try again."
                    st.error(msg)
                    add_message("assistant", msg)

        st.rerun()

# ── Phase: done (follow-up questions) ────────────────────────────────────────

elif st.session_state.phase == "done":
    if prompt := st.chat_input("Ask a follow-up question about this repo or issue..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_query(st.session_state.selected_repo, prompt)

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
    st.markdown("### Current Session")

    if st.session_state.skills:
        st.markdown(f"**Skills:** {', '.join(st.session_state.skills)}")
    if st.session_state.selected_repo:
        st.markdown(f"**Repo:** `{st.session_state.selected_repo}`")
    if st.session_state.selected_issue:
        st.markdown(f"**Issue:** #{st.session_state.selected_issue}")

    if not st.session_state.skills:
        st.caption("No active session yet.")

    st.markdown("---")

    if st.button("New Conversation", use_container_width=True, type="primary"):
        # Keep keys — just reset conversation state
        keys_to_keep = {
            "anthropic_key": st.session_state.anthropic_key,
            "openai_key": st.session_state.openai_key,
            "github_pat": st.session_state.github_pat,
            "admin_passphrase": st.session_state.admin_passphrase,
        }
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.session_state.update(keys_to_keep)
        st.session_state.phase = "skills"  # skip setup on reset
        st.rerun()
