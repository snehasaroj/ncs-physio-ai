"""Shared Streamlit helpers: session state, data loading, small UI pieces."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ncs_schema import DEFAULT_NORMALS

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_cohort.csv"


@st.cache_data
def load_demo() -> pd.DataFrame:
    if DEMO.exists():
        return pd.read_csv(DEMO)
    from ncs_synthetic import make_cohort
    return make_cohort()


def init_state() -> None:
    if "cohort" not in st.session_state:
        st.session_state.cohort = load_demo().copy()
        st.session_state.data_source = "Synthetic demo cohort (150 cases, no real patients)"
    st.session_state.setdefault("normals", dict(DEFAULT_NORMALS))
    st.session_state.setdefault("settings", {
        "institution": "Rajiv Gandhi Medical College, Thane",
        "department": "Department of Physiology - Neurophysiology Laboratory",
        "verified_by": "",
    })
    with st.sidebar:
        st.caption(f"**Data:** {st.session_state.data_source}")
        st.caption(f"Records loaded: {len(st.session_state.cohort)}")


def record(row: pd.Series | dict) -> dict:
    """pandas row -> plain dict with NaN replaced by None."""
    if isinstance(row, pd.Series):
        return row.where(row.notna(), None).to_dict()
    return dict(row)


def case_picker(label: str = "Select case", key: str = "case") -> dict | None:
    df = st.session_state.cohort
    if df.empty:
        st.info("No records yet.")
        return None
    def fmt(i):
        r = df.loc[i]
        gt = f" - truth: {r['ground_truth']}" if "ground_truth" in df.columns and pd.notna(r.get("ground_truth")) else ""
        return f"{r['study_id']} ({r.get('age', '')}{str(r.get('sex', ''))[:1]}){gt}"
    idx = st.selectbox(label, df.index, format_func=fmt, key=key)
    return record(df.loc[idx])
