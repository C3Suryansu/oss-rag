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

repo_url = st.text_input(
    "Github Repository URL",
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