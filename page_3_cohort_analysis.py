import pandas as pd
import plotly.express as px
import streamlit as st

from common import init_state, load_demo
import ncs_analysis as analysis
from ncs_schema import STUDIES

init_state()
st.title("📊 Cohort analysis")

with st.expander("Data source", expanded=False):
    up = st.file_uploader("Upload your own CSV with the same columns (see demo_cohort.csv)", type="csv")
    c1, c2 = st.columns(2)
    if up is not None and c1.button("Use uploaded file"):
        st.session_state.cohort = pd.read_csv(up)
        st.session_state.data_source = f"Uploaded: {up.name}"
        st.rerun()
    if c2.button("Reset to synthetic demo cohort"):
        st.session_state.cohort = load_demo().copy()
        st.session_state.data_source = "Synthetic demo cohort (150 cases, no real patients)"
        st.rerun()
    st.caption("Uploaded data stays in your browser session only. Do not upload identifiable patient data to a public deployment.")

t = analysis.analysis_table(st.session_state.cohort, st.session_state.normals)
st.download_button("⬇️ Download analysis-ready CSV (for SPSS / R)", t.to_csv(index=False).encode(),
                   "ncs_analysis_table.csv", "text/csv")

m = st.columns(5)
m[0].metric("Patients", len(t))
m[1].metric("Age, mean ± SD", f"{t['age'].mean():.1f} ± {t['age'].std():.1f}")
m[2].metric("Female", f"{(t['sex'] == 'Female').mean() * 100:.0f}%")
m[3].metric("NDI %, median", f"{t['ndi_percent'].median():.0f}")
m[4].metric("Abnormal NCS", f"{(t['ncs_result'] == 'Abnormal').mean() * 100:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Physiology plots", "Correlations", "Group comparisons"])
with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(t, x="ndi_percent", color="ndi_severity", nbins=20, title="Neck Disability Index (%)"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(t["ncs_pattern"].value_counts().reset_index(), x="ncs_pattern", y="count",
                           title="Electrodiagnostic pattern (rule engine)"), use_container_width=True)
    c1.plotly_chart(px.histogram(t, x="ipaq_category", color="sex", barmode="group", title="IPAQ category",
                                 category_orders={"ipaq_category": ["Low", "Moderate", "High"]}), use_container_width=True)
    c2.plotly_chart(px.histogram(t, x="bmi_category", title="BMI (WHO Asia-Pacific)",
                                 category_orders={"bmi_category": ["Underweight", "Normal", "Overweight", "Obese"]}),
                    use_container_width=True)
with tab2:
    st.markdown("F-wave latency depends on the length of the pathway, so it should rise with height. "
                "This is a useful check on the data before testing any disease hypothesis.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(t, x="height_cm", y="sym_ulnM_f", color="ncs_pattern", trendline=None,
                               labels={"sym_ulnM_f": "Ulnar F-wave latency, symptomatic side (ms)"},
                               title="Height vs ulnar F-wave latency"), use_container_width=True)
    c2.plotly_chart(px.box(t, x="ncs_pattern", y="sym_medS_lat", points="all",
                           labels={"sym_medS_lat": "Median sensory peak latency (ms)"},
                           title="Median sensory latency by pattern"), use_container_width=True)
    param = st.selectbox("Parameter (symptomatic side)", [f"sym_{s.key}_{p.key}" for s in STUDIES for p in s.params])
    grp = st.selectbox("Group by", ["ndi_severity", "ipaq_category", "bmi_category", "sex", "ncs_pattern"])
    st.plotly_chart(px.box(t, x=grp, y=param, points="all"), use_container_width=True)
with tab3:
    xs = st.multiselect("Clinical variables", ["ndi_percent", "vas", "pain_duration_months", "age", "height_cm", "bmi",
                                               "bc_fat_pct", "bc_muscle_kg", "bc_visceral", "ipaq_total_met", "ipaq_sit_min"],
                        default=["ndi_percent", "vas", "bmi", "bc_fat_pct", "ipaq_total_met", "height_cm"])
    ys = st.multiselect("NCS variables (symptomatic side)", [f"sym_{s.key}_{p.key}" for s in STUDIES for p in s.params],
                        default=["sym_medS_lat", "sym_medS_amp", "sym_ulnM_f", "sym_medM_f", "sym_ulnM_amp"])
    res = analysis.correlations(t, xs, ys)
    if not res.empty:
        st.dataframe(res.round({"rho": 3, "p": 4, "p_fdr": 4}), hide_index=True,
                     use_container_width=True)
        piv = res.pivot(index="x", columns="y", values="rho")
        st.plotly_chart(px.imshow(piv, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
                                  title="Spearman rho"), use_container_width=True)
        st.caption("p_fdr = Benjamini-Hochberg adjusted p-value across all the correlations shown.")
with tab4:
    grp = st.selectbox("Grouping variable", ["ipaq_category", "ndi_severity", "sex", "bmi_category", "ncs_result"], key="g2")
    vals = st.multiselect("Compare", [f"sym_{s.key}_{p.key}" for s in STUDIES for p in s.params] +
                          ["ndi_percent", "vas", "bc_fat_pct"], default=["sym_medS_lat", "sym_ulnM_f", "ndi_percent"])
    cmp = analysis.compare_groups(t, grp, vals)
    if not cmp.empty:
        st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.markdown("**Chi-square: NCS result vs NDI severity**")
    ct, chi = analysis.chi_square(t, "ncs_result", "ndi_severity")
    st.dataframe(ct, use_container_width=True)
    if chi:
        st.write(f"χ² = {chi['chi2']:.2f}, df = {chi['dof']}, p = {chi['p']:.4f}")
