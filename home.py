import streamlit as st

from common import init_state

init_state()

st.title("NCS-Physio AI")
st.subheader("Electrophysiological assessment of cervical pain, with LLM + RAG interpretation and an AI-evaluation benchmark")

st.markdown("""
This app grew out of my MD (Physiology) thesis, **"Electrophysiological Assessment of Cervical Pain: A Cross-sectional
Study"** (Rajiv Gandhi Medical College, Thane; guide: Dr. Swapnali Kadam). In the study I assess the median, ulnar and
radial nerves (latency, amplitude, conduction velocity and F-wave latency) in 150+ patients, together with the
**Neck Disability Index**, the **IPAQ** and **body composition analysis**.

The public version runs on a **synthetic cohort** that I generated from the underlying physiology. It contains no real
patient data, and because the correct diagnosis of every case is known, it doubles as a benchmark.
""")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🩺 Clinical workflow")
    st.markdown("- **Patient entry**: NCS, NDI, IPAQ, body composition\n"
                "- Rule-based interpretation you can audit\n- A4 **PDF report**")
with c2:
    st.markdown("#### 📊 Data analysis")
    st.markdown("- Analysis-ready table with derived scores\n- Spearman correlations, Mann-Whitney / Kruskal-Wallis, "
                "chi-square, **Benjamini-Hochberg FDR** correction\n- Interactive Plotly charts")
with c3:
    st.markdown("#### 🤖 LLM, RAG and AI evaluation")
    st.markdown("- **RAG**: a BM25 retriever over a curated electrophysiology knowledge base, queried from the case's "
                "abnormal findings\n- An LLM (Gemini / OpenAI / Claude) writes the interpretation with citations\n"
                "- An **automatic critic** checks it: wrong pattern, missed or invented abnormalities, "
                "hallucinated numbers, invalid citations, physiology errors\n"
                "- A **benchmark** compares RAG against no-RAG and against rule-based and trivial baselines")

st.divider()
st.markdown("""
**How the interpretation logic works (physiology first):**
amplitude reflects the number of conducting axons; latency and velocity reflect myelin. Cervical root lesions are
*pre-ganglionic*: the dorsal root ganglion is spared, so **SNAPs stay normal** even with sensory symptoms. A low SNAP
therefore points away from radiculopathy, towards an entrapment, a plexopathy or a polyneuropathy. Each rule in
`ncs_interpret.py` is written so that a physiologist can check it.

Use the pages in the sidebar. To try the AI pages, add an API key under *Settings → Secrets* (a free Google Gemini key
works).
""")
st.caption("Research and education tool. Not for clinical decision-making without verification by a qualified clinician.")
