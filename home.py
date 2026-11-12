import streamlit as st

from common import init_state

init_state()

st.title("NCS-Physio AI")
st.write("A tool for recording, reporting and analysing upper-limb nerve conduction studies, with AI-assisted interpretation.")

st.subheader("What it can do")
st.markdown("""
- Record nerve conduction findings for the median, ulnar and radial nerves, along with the Neck Disability Index, IPAQ and body composition
- Flag abnormal values and suggest an interpretation
- Generate a PDF report for each patient
- Analyse a whole group of patients with charts and basic statistics
- Compare AI-generated interpretations with the measured data
""")

st.subheader("Who can benefit")
st.markdown("""
- **Physiology and neurology residents** working on NCS-based research
- **Neurophysiology labs** that need structured reporting
- **Teachers and students** learning how NCS findings are interpreted
- **Researchers** studying how accurately AI interprets clinical data
""")

st.caption("Demo data is synthetic and contains no real patient information. For research and teaching only, not for clinical diagnosis.")
