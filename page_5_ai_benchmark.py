import pandas as pd
import plotly.express as px
import streamlit as st

from common import init_state, load_demo
import ncs_benchmark as benchmark
import ncs_llm as llm

init_state()
st.title("🧪 Benchmark: can an LLM interpret nerve conduction studies?")
st.markdown("""
**Design.** The cases come from the synthetic cohort, where the correct electrodiagnostic pattern is known by
construction. Every output is scored on:
- **Pattern accuracy** and **side accuracy**
- **Abnormal-value F1**: whether it found every out-of-range parameter and invented none
- **Unsupported numbers**: values quoted in the narrative that appear nowhere in the data (hallucination)
- **Physiology errors**: e.g. calling a case radiculopathy while a SNAP is reduced

The **RAG vs no-RAG** ablation isolates what retrieval adds. The rule engine (upper bound) and *always normal*
(lower bound) give the scores context.
""")

cohort = load_demo()
c1, c2, c3 = st.columns(3)
n = c1.slider("Number of cases (stratified by pattern)", 6, 60, 12, step=6)
seed = c2.number_input("Sampling seed", value=0, step=1)
conds = c3.multiselect("Conditions", ["rag", "no_rag"], default=["rag", "no_rag"])
cases = benchmark.select_cases(cohort, n, int(seed))
st.caption("Case mix: " + ", ".join(f"{k} ({v})" for k, v in cases["ground_truth"].value_counts().items()))

base = benchmark.run_baselines(cases)
providers = llm.available_providers()
if providers:
    p1, p2 = st.columns(2)
    provider = p1.selectbox("Provider", providers)
    model = p2.text_input("Model", llm.DEFAULT_MODELS[provider])
    if st.button(f"Run benchmark ({n * len(conds)} LLM calls)", type="primary"):
        bar = st.progress(0.0)
        st.session_state.bench = benchmark.run_llm(cases, provider, model, conds, progress=bar.progress)
else:
    st.info("Add an API key in the app's Secrets to benchmark an LLM. The baselines are shown below.")

results = pd.concat([base, st.session_state.get("bench", pd.DataFrame())], ignore_index=True)
summary = benchmark.summarise(results)
summary["run"] = summary["system"] + " / " + summary["condition"]
st.subheader("Summary")
st.dataframe(summary.drop(columns="run").round(3), hide_index=True, use_container_width=True)
long = summary.melt(id_vars="run", value_vars=["pattern_accuracy", "side_accuracy", "param_F1"])
st.plotly_chart(px.bar(long, x="variable", y="value", color="run", barmode="group", range_y=[0, 1]),
                use_container_width=True)

if "bench" in st.session_state:
    b = st.session_state.bench
    ok = b[b["error"].fillna("") == ""] if "error" in b.columns else b
    if not ok.empty:
        st.subheader("Confusion matrix (LLM)")
        cond = st.selectbox("Condition", ok["condition"].unique())
        sub = ok[ok["condition"] == cond]
        cm = pd.crosstab(sub["ground_truth"], sub["predicted"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  labels={"x": "Predicted", "y": "Ground truth"}), use_container_width=True)
    st.subheader("Per-case results")
    st.dataframe(b, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download results CSV", b.to_csv(index=False).encode(), "benchmark_results.csv", "text/csv")
