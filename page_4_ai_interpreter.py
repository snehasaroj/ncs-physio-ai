import json

import streamlit as st

from common import case_picker, init_state
import ncs_ai as ai
import ncs_llm as llm
from ncs_interpret import PATTERNS

init_state()
st.title("🤖 AI interpreter: LLM + RAG, checked by an automatic critic")
st.markdown("Pick a case. The retriever pulls the most relevant physiology passages based on the case's **abnormal "
            "findings**, the LLM writes a structured interpretation with citations, and the critic checks every claim "
            "against the measured data.")

providers = llm.available_providers()
with st.sidebar:
    st.subheader("Model")
    if providers:
        provider = st.selectbox("Provider", providers)
        model = st.text_input("Model", llm.DEFAULT_MODELS[provider])
    else:
        provider, model = None, None
        st.warning("No API key configured. Add `GEMINI_API_KEY` (free at aistudio.google.com) in the app's Secrets.")
    use_rag = st.toggle("Use RAG (retrieved passages)", True)
    k = st.slider("Passages to retrieve (k)", 1, 8, 4)

rec = case_picker()
question = st.text_input("Optional clinician question", placeholder="e.g. Is this consistent with C8 radiculopathy?")
if not rec:
    st.stop()

prompt, hits, query, truth = ai.build_prompt(rec, st.session_state.normals, use_rag, question, k)

with st.expander(f"🔎 Retrieved passages ({len(hits)})", expanded=use_rag):
    st.caption(f"Query built from the case data: *{query}*")
    for i, (c, score) in enumerate(hits, 1):
        st.markdown(f"**[{i}] {c.title}**  ·  BM25 = {score:.2f}")
        st.write(c.text)
with st.expander("🧾 Full prompt sent to the model"):
    st.code(prompt, language="text")

col_rule, col_ai = st.columns(2)
with col_rule:
    st.subheader("Rule engine (reference)")
    st.metric("Pattern", PATTERNS.get(truth.primary, truth.primary))
    st.write(f"Side: **{truth.primary_side}**")
    if rec.get("ground_truth"):
        st.caption(f"Synthetic ground truth: {rec['ground_truth']} / {rec.get('ground_truth_side')}")
    st.text(truth.impression())

with col_ai:
    st.subheader("LLM interpretation")
    run = st.button("Run LLM", type="primary", disabled=provider is None)
    if run:
        with st.spinner("Calling model…"):
            try:
                res = ai.run_case(rec, provider, model, st.session_state.normals, use_rag, question, k)
                st.session_state.ai_last = (rec["study_id"], use_rag, res)
            except Exception as e:  # show any provider / parsing error to the user
                st.error(str(e))
    last = st.session_state.get("ai_last")
    if last and last[0] == rec["study_id"]:
        res = last[2]
        out = res["output"]
        st.metric("Pattern", PATTERNS.get(out.get("primary_pattern"), str(out.get("primary_pattern"))))
        st.write(f"Side: **{out.get('side')}**  ·  confidence {out.get('confidence')}")
        st.write(out.get("interpretation", ""))
        if out.get("differential"):
            st.markdown("**Differential:** " + "; ".join(map(str, out["differential"])))
        with st.expander("Raw JSON"):
            st.code(json.dumps(out, indent=2), language="json")

        st.subheader("🧪 Critique")
        c = ai.critique(out, rec, res["truth"], st.session_state.normals, len(res["hits"]),
                        rec.get("ground_truth"), rec.get("ground_truth_side"))
        m = st.columns(4)
        m[0].metric("Pattern", "✅" if c.pattern_correct else "❌")
        m[1].metric("Side", "✅" if c.side_correct else "❌")
        m[2].metric("Abnormal-value F1", f"{c.f1:.2f}")
        m[3].metric("Issues", c.issues)
        for lab, items in [("Missed abnormalities", c.missed), ("Invented abnormalities", c.invented),
                           ("Numbers not found in the data (possible hallucination)", c.unsupported_numbers),
                           ("Invalid citations", c.bad_citations), ("Physiology errors", c.physiology_flags)]:
            if items:
                st.warning(f"**{lab}:** " + ", ".join(items))
        if not c.issues:
            st.success("No issues detected by the critic.")
