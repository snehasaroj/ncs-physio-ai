"""Entry point. All files live at the top level of the repository, and pages are registered here."""
import streamlit as st

st.set_page_config(page_title="NCS-Physio AI", page_icon="🧠", layout="wide")

pg = st.navigation([
    st.Page("home.py", title="Home", icon="🧠", default=True),
    st.Page("page_1_patient_entry.py", title="Patient entry", icon="🩺"),
    st.Page("page_2_reports.py", title="Reports", icon="📄"),
    st.Page("page_3_cohort_analysis.py", title="Cohort analysis", icon="📊"),
    st.Page("page_4_ai_interpreter.py", title="AI interpreter (RAG)", icon="🤖"),
    st.Page("page_5_ai_benchmark.py", title="AI benchmark", icon="🧪"),
    st.Page("page_6_knowledge_base.py", title="Knowledge base", icon="📚"),
])
pg.run()
