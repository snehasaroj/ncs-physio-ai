import pandas as pd
import streamlit as st

from common import case_picker, init_state
from ncs_interpret import interpret, is_abnormal
from ncs_report import build_pdf
from ncs_schema import SIDES, STUDIES
from ncs_scoring import derived, to_num

init_state()
st.title("📄 Reports")

N = st.session_state.normals
rec = case_picker()
if rec:
    it = interpret(rec, N)
    d = derived(rec)
    left, right = st.columns([3, 2])
    with left:
        rows = []
        for s in STUDIES:
            for p in s.params:
                row = {"Study": s.name, "Parameter": f"{p.label} ({p.unit})",
                       "Normal": f"{'≤' if p.direction == 'max' else '≥'}{N[f'{s.key}_{p.key}']:g}"}
                for sd, side in SIDES.items():
                    v = to_num(rec.get(f"{sd}_{s.key}_{p.key}"))
                    row[side] = None if v is None else v
                    row[f"{side} flag"] = "⚠️" if is_abnormal(s.key, p.key, v, N) else ""
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=620)
    with right:
        st.metric("Rule-engine pattern", it.primary.replace("_", " "))
        st.write(f"NDI: **{d['ndi_percent']:.0f}% ({d['ndi_severity']})**" if d["ndi_percent"] is not None else "NDI: -")
        st.write(f"IPAQ: **{d['ipaq_category']}** ({d['ipaq_total_met']:.0f} MET-min/wk)" if d["ipaq_total_met"] is not None else "IPAQ: -")
        imp = st.text_area("Impression", it.impression(), height=280)
        with st.expander("Report header"):
            for k, lab in [("institution", "Institution"), ("department", "Department"), ("verified_by", "Verified by")]:
                st.session_state.settings[k] = st.text_input(lab, st.session_state.settings.get(k, ""))
        st.download_button("⬇️ Download PDF report", build_pdf(rec, st.session_state.settings, N, imp),
                           file_name=f"NCS_{rec['study_id']}.pdf", mime="application/pdf", type="primary")
