import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Academic City RAG Chatbot", layout="wide")
st.title("Academic City RAG Chatbot")
st.caption("Data: Ghana Election Results CSV + 2025 Budget Statement PDF")

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

query = st.text_input("Enter your question")
col1, col2 = st.columns(2)
use_hybrid = col1.checkbox("Use hybrid retrieval fix", value=True)
top_k = col2.slider("Top-k", min_value=2, max_value=10, value=5)

if st.button("Ask") and query.strip():
    result = pipeline.answer(query=query, top_k=top_k, use_hybrid=use_hybrid)

    st.subheader("Final Response")
    st.write(result["response"])

    st.subheader("Retrieved Chunks")
    for i, d in enumerate(result["retrieved_docs"], start=1):
        with st.expander(f"Chunk {i} | score={d['score']:.4f} | source={d['metadata'].get('source')}"):
            st.write(d["text"])
            st.json(d["metadata"])

    st.subheader("Prompt sent to LLM")
    st.code(result["final_prompt"], language="markdown")
