import streamlit as st

from common import init_state
import ncs_rag as rag

init_state()
st.title("📚 Knowledge base and retriever")
st.markdown("These are the passages the RAG pipeline retrieves from. I wrote them myself, as short, citable summaries "
            "of standard electrodiagnostic physiology, in `kb_*.md`. Search below to see the BM25 retriever "
            "at work.")

q = st.text_input("Search", "SNAP preserved in cervical radiculopathy")
for i, (c, s) in enumerate(rag.index().search(q, k=5), 1):
    with st.container(border=True):
        st.markdown(f"**{i}. {c.title}**  ·  score {s:.2f}")
        st.write(c.text)

with st.expander(f"All {len(rag.index().chunks)} chunks"):
    for c in rag.index().chunks:
        st.markdown(f"- **{c.title}** ({len(c.text.split())} words)")
