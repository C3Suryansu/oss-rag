import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="OSS Onboarding Assistant",
    page_icon="🚀",
    layout="centered"
)

st.title("OSS Onboarding Assistant")
st.markdown("Ask anything about any open source repository. Get beginner-friendly answers instantly.")

with st.form("query_form"):
    repo_url = st.text_input(
        "Github Repository URL",
        placeholder="https://github.com/kubeflow/pipelines"
    )
    question = st.text_area(
        "Your Question",
        placeholder="How do I set up this project locally?",
        height = 100
    )
    submitted=st.form_submit_button("Ask", use_container_width=True)

    if submitted:
        if not repo_url or not question:
            st.error("Please provide both a repository URL and question")
        else:
            with st.spinner("Fetching repo data and generating answer...."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"repo_url":repo_url, "question":question}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Answer generated!")
                        st.markdown("### Answer")
                        st.markdown(data["answer"])
                        st.markdown("### Sources")
                        for source in set(data["sources"]):
                            st.badge(source)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

