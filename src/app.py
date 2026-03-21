import streamlit as st
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OSS Onboarding Assistant",
    page_icon="🚀",
    layout="centered"
)

st.title("OSS Onboarding Assistant")
st.markdown("Your AI-powered guide to open source contribution.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🔍 Find Repos for My Skills",
     "Analyze Issues", 
     "💬 Ask About a Repo", 
     "Deep Dive Issue", 
     "Navigate the Codebase for deeper issuedive",
     "Complete contribution Agent"]
    )

# --- Tab 1: Skill Matching ---
with tab1:
    st.markdown("### Find OSS Repos Matching Your Skills")
    skills_input = st.text_input("Your Skills", placeholder="python, machine learning, pytorch")
    find_submitted = st.button("Find Repos", use_container_width=True)

    if find_submitted:
        if not skills_input:
            st.error("Please enter at least one skill")
        else:
            skills = [s.strip() for s in skills_input.split(",")]
            with st.spinner("Finding and ranking repos... this takes 2-3 minutes"):
                try:
                    response = requests.post(
                        f"{API_URL}/skill-match",
                        json={"skills": skills}
                    )
                    if response.status_code == 200:
                        st.markdown("### Recommended Repos")
                        st.markdown(response.json()["result"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
                    

# --- Tab 2: Find the best issues to work on
with tab2:
    st.markdown("### Find the Right Issues to Work On")
    st.markdown("Enter a repo and your skills — get top 2-3 issues ranked by beginner-friendliness.")

    repo_input = st.text_input(
        "GitHub Repo (owner/repo)",
        placeholder="pytorch/pytorch"
    )
    skills_input2 = st.text_input(
        "Your Skills",
        placeholder="python, machine learning, pytorch",
        key="skills2"
    )
    analyze_submitted = st.button("Analyze Issues", use_container_width=True)

    if analyze_submitted:
        if not repo_input or not skills_input2:
            st.error("Please provide both a repo and your skills")
        else:
            skills = [s.strip() for s in skills_input2.split(",")]
            with st.spinner("Analyzing issues... this takes 2-3 minutes"):
                try:
                    response = requests.post(
                        f"{API_URL}/analyze-issues",
                        json={"repo_full_name": repo_input, "skills": skills}
                    )
                    if response.status_code == 200:
                        st.markdown("### Recommended Issues")
                        st.markdown(response.json()["result"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

# --- Tab 3: Ask About a Repo ---
with tab3:
    st.markdown("### Ask Anything About a Repository")

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/kubeflow/pipelines"
    )
    question = st.text_area(
        "Your Question",
        placeholder="How do I set up this project locally?",
        height=100
    )
    submitted = st.button("Ask", use_container_width=True)

    if submitted:
        if not repo_url or not question:
            st.error("Please provide both a repository URL and question")
        else:
            st.markdown("### Answer")

            def stream_response():
                with requests.post(
                    f"{API_URL}/query/stream",
                    json={"repo_url": repo_url, "question": question},
                    stream=True
                ) as response:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk

            try:
                st.write_stream(stream_response())
            except Exception as e:
                st.error(f"Could not connect to API: {e}")
# --- Tab 4: Deep Dive into an issue
with tab4:
    st.markdown("### Deep Dive on a Specific Issue")
    st.markdown("Paste a repo and issue number — get a full breakdown with reproduction steps and file pointers.")

    repo_dd = st.text_input("GitHub Repo (owner/repo)", placeholder="pytorch/pytorch", key="repo_dd")
    issue_num = st.number_input("Issue Number", min_value=1, step=1)
    dd_submitted = st.button("Deep Dive", use_container_width=True)

    if dd_submitted:
        if not repo_dd:
            st.error("Please provide a repo")
        else:
            with st.spinner("Analyzing issue... this takes 2-3 minutes"):
                try:
                    response = requests.post(
                        f"{API_URL}/deepdive-issue",
                        json={"repo_full_name": repo_dd, "issue_number": int(issue_num)}
                    )
                    if response.status_code == 200:
                        st.markdown("### Issue Deep-Dive")
                        st.markdown(response.json()["result"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
# --- Tab 5: Navigate the codebase
with tab5:
    st.markdown("### Navigate the Codebase")
    st.markdown("Give a repo and file paths — ask questions about the code.")

    repo_nav = st.text_input("GitHub Repo (owner/repo)", placeholder="scikit-learn/scikit-learn", key="repo_nav")
    files_nav = st.text_input("File Paths (comma separated)", placeholder="sklearn/utils/validation.py", key="files_nav")
    question_nav = st.text_area("Your Question", placeholder="Where should I add input validation for a new parameter?", height=100, key="question_nav")
    nav_submitted = st.button("Navigate", use_container_width=True)

    if nav_submitted:
        if not repo_nav or not files_nav or not question_nav:
            st.error("Please fill in all fields")
        else:
            file_paths = [f.strip() for f in files_nav.split(",")]
            with st.spinner("Navigating codebase..."):
                try:
                    response = requests.post(
                        f"{API_URL}/navigate-codebase",
                        json={
                            "repo_full_name": repo_nav,
                            "file_paths": file_paths,
                            "question": question_nav
                        }
                    )
                    if response.status_code == 200:
                        st.markdown("### Code Navigation Result")
                        st.markdown(response.json()["result"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")
# --- Tab 6: Complete contribution agent
with tab6:
    st.markdown("### Full Contribution Guide")
    st.markdown("Give a repo, issue number, and your skills — get a complete contribution guide.")

    repo_ca = st.text_input("GitHub Repo (owner/repo)", placeholder="scikit-learn/scikit-learn", key="repo_ca")
    issue_ca = st.number_input("Issue Number", min_value=1, step=1, key="issue_ca")
    skills_ca = st.text_input("Your Skills", placeholder="python, machine learning", key="skills_ca")
    question_ca = st.text_area("Specific Question (optional)", placeholder="Where should I add the fix?", height=80, key="question_ca")
    ca_submitted = st.button("Get Contribution Guide", use_container_width=True)

    if ca_submitted:
        if not repo_ca or not skills_ca:
            st.error("Please provide repo and skills")
        else:
            skills = [s.strip() for s in skills_ca.split(",")]
            with st.spinner("Running contribution agent... this takes 3-5 minutes"):
                try:
                    response = requests.post(
                        f"{API_URL}/contribution-agent",
                        json={
                            "skills": skills,
                            "selected_repo": repo_ca,
                            "selected_issue": int(issue_ca),
                            "question": question_ca or None
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown("### Deep Dive")
                        st.markdown(data["deepdive"])
                        st.markdown("### Code Navigation")
                        st.markdown(data["navigation"])
                        if data["file_paths"]:
                            st.markdown("### Relevant Files")
                            for f in data["file_paths"]:
                                st.code(f)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")