import pandas as pd
import streamlit as st

from common import init_state
from ncs_interpret import interpret
from ncs_report import build_pdf
from ncs_schema import NDI_SECTIONS, SIDES, STUDIES
from ncs_scoring import derived

init_state()
st.title("🩺 Patient entry")
st.caption("Enter one patient's data. Derived scores and the interpretation update when you press **Save**.")

N = st.session_state.normals


with st.form("entry", clear_on_submit=False):
    t1, t2, t3, t4, t5 = st.tabs(["Patient & clinical", "Body composition", "NDI", "IPAQ", "Nerve conduction"])
    rec: dict = {}
    with t1:
        c = st.columns(4)
        rec["study_id"] = c[0].text_input("Study ID *")
        rec["age"] = c[1].number_input("Age (years)", value=None, min_value=0, max_value=110, step=1)
        rec["sex"] = c[2].selectbox("Sex", ["Male", "Female", "Other"], index=None)
        rec["date"] = str(c[3].date_input("Date of test"))
        c = st.columns(4)
        rec["height_cm"] = c[0].number_input("Height (cm)", value=None, min_value=0.0)
        rec["weight_kg"] = c[1].number_input("Weight (kg)", value=None, min_value=0.0)
        rec["occupation"] = c[2].text_input("Occupation")
        rec["limb_temp_c"] = c[3].number_input("Limb skin temperature (°C)", value=None)
        c = st.columns(4)
        rec["pain_side"] = c[0].selectbox("Side of neck pain / radiation", ["Right", "Left", "Bilateral", "Central"], index=None)
        rec["pain_duration_months"] = c[1].number_input("Duration (months)", value=None, min_value=0.0)
        rec["vas"] = c[2].number_input("Pain VAS (0-10)", value=None, min_value=0.0, max_value=10.0)
        rec["paraesthesia"] = c[3].selectbox("Paraesthesia", ["None", "Right", "Left", "Bilateral"], index=None)
    with t2:
        c = st.columns(3)
        for i, (k, lab) in enumerate([("bc_fat_pct", "Body fat (%)"), ("bc_muscle_kg", "Skeletal muscle (kg)"),
                                      ("bc_visceral", "Visceral fat level"), ("bc_ffm_kg", "Fat-free mass (kg)"),
                                      ("bc_tbw_l", "Total body water (L)"), ("bc_bmr_kcal", "BMR (kcal)")]):
            rec[k] = c[i % 3].number_input(lab, value=None, min_value=0.0)
    with t3:
        st.caption("Score each section 0-5. Leave a section blank if it was not applicable, e.g. not driving.")
        c = st.columns(5)
        for i, name in enumerate(NDI_SECTIONS):
            rec[f"ndi_q{i + 1}"] = c[i % 5].selectbox(f"{i + 1}. {name}", [0, 1, 2, 3, 4, 5], index=None)
    with t4:
        c = st.columns(2)
        for i, (k, lab) in enumerate([("ipaq_vig_days", "Vigorous: days/week"), ("ipaq_vig_min", "Vigorous: min/day"),
                                      ("ipaq_mod_days", "Moderate: days/week"), ("ipaq_mod_min", "Moderate: min/day"),
                                      ("ipaq_walk_days", "Walking: days/week"), ("ipaq_walk_min", "Walking: min/day"),
                                      ("ipaq_sit_min", "Sitting: min/weekday")]):
            rec[k] = c[i % 2].number_input(lab, value=None, min_value=0, step=1)
    with t5:
        for s in STUDIES:
            st.markdown(f"**{s.name}** <span style='color:gray;font-size:0.85em'>{s.recording}</span>", unsafe_allow_html=True)
            cols = st.columns(len(s.params) * 2)
            j = 0
            for p in s.params:
                for sd, side in SIDES.items():
                    op = "≤" if p.direction == "max" else "≥"
                    rec[f"{sd}_{s.key}_{p.key}"] = cols[j].number_input(
                        f"{side[0]} {p.label} ({p.unit}) {op}{N[f'{s.key}_{p.key}']:g}", value=None,
                        key=f"{sd}_{s.key}_{p.key}", format="%.2f")
                    j += 1
    submitted = st.form_submit_button("Save & interpret", type="primary")

if submitted:
    if not rec["study_id"]:
        st.error("Study ID is required.")
        st.stop()
    df = st.session_state.cohort
    df = df[df["study_id"] != rec["study_id"]]
    st.session_state.cohort = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    st.session_state.last_rec = rec
    st.success(f"Saved {rec['study_id']} to this session. Download the dataset from *Cohort analysis* to keep it.")

rec = st.session_state.get("last_rec")
if rec:
    it = interpret(rec, N)
    d = derived(rec)
    m = st.columns(4)
    m[0].metric("BMI", f"{d['bmi']:.1f}" if d["bmi"] else "-", d["bmi_category"] or None, delta_color="off")
    m[1].metric("NDI", f"{d['ndi_percent']:.0f}%" if d["ndi_percent"] is not None else "-", d["ndi_severity"] or None, delta_color="off")
    m[2].metric("IPAQ", f"{d['ipaq_total_met']:.0f} MET-min/wk" if d["ipaq_total_met"] is not None else "-", d["ipaq_category"] or None, delta_color="off")
    m[3].metric("Abnormal NCS values", len(it.abnormal))
    if it.abnormal:
        st.markdown("**Abnormal findings**\n" + "\n".join(f"- {a.describe()}" for a in it.abnormal))
    imp = st.text_area("Impression (you can edit this before downloading)", it.impression(), height=200)
    st.download_button("⬇️ Download PDF report", build_pdf(rec, st.session_state.settings, N, imp),
                       file_name=f"NCS_{rec['study_id']}.pdf", mime="application/pdf")
