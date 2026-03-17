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

tab1, tab2 = st.tabs(["🔍 Find Repos for My Skills", "💬 Ask About a Repo"])

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
                    
# --- Tab 2: Ask About a Repo ---
with tab2:
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