import json

import pandas as pd
import pytest

import ncs_ai as ai
import ncs_benchmark as benchmark
import ncs_rag as rag
from ncs_interpret import interpret
from ncs_scoring import bmi, bmi_category, ipaq, ndi
from ncs_synthetic import make_cohort


def test_bmi_asia_pacific():
    assert bmi(165, 70) == pytest.approx(25.71, abs=0.01)
    assert bmi_category(22.9) == "Normal" and bmi_category(23.0) == "Overweight"


def test_ndi_handles_missing_section():
    rec = {f"ndi_q{i}": 2 for i in range(1, 11)}
    rec["ndi_q8"] = None
    out = ndi(rec)
    assert out["answered"] == 9 and out["percent"] == pytest.approx(40.0)
    assert out["severity"] == "Moderate"  # 20/50 equivalent


def test_ipaq_truncation_and_category():
    rec = {"ipaq_vig_days": 0, "ipaq_mod_days": 3, "ipaq_mod_min": 40, "ipaq_walk_days": 5, "ipaq_walk_min": 30}
    out = ipaq(rec)
    assert out["total_met"] == pytest.approx(3.3 * 30 * 5 + 4 * 40 * 3)
    assert out["category"] == "Moderate"
    assert ipaq({"ipaq_walk_days": 7, "ipaq_walk_min": 5})["total_met"] == 0  # <10 min bouts excluded


def _normal_case():
    vals = {"medM": (3.4, 9, 57, 26), "ulnM": (2.6, 10, 60, 27), "radM": (2.2, 5, 60, None),
            "medS": (2.9, 40, 57), "ulnS": (2.6, 32, 57), "radS": (2.2, 28, 57)}
    rec = {}
    for sd in "RL":
        for k, v in vals.items():
            for p, x in zip(("lat", "amp", "cv", "f"), v):
                if x is not None:
                    rec[f"{sd}_{k}_{p}"] = x
    return rec


def test_normal_and_cts_and_radiculopathy():
    rec = _normal_case()
    assert interpret(rec).primary == "normal"
    cts = {**rec, "R_medS_lat": 4.2, "R_medS_cv": 42}
    assert interpret(cts).primary == "median_neuropathy_wrist" and interpret(cts).primary_side == "R"
    rad = {**rec, "L_ulnM_f": 34, "L_ulnM_amp": 4.0}
    assert interpret(rad).primary == "preganglionic_radiculopathy"
    # Same motor change but with a low ulnar SNAP -> post-ganglionic, NOT radiculopathy
    post = {**rad, "L_ulnS_amp": 8}
    assert interpret(post).primary == "postganglionic_lesion"


def test_synthetic_ground_truth_is_consistent():
    df = make_cohort(60, seed=1)
    for _, r in df.iterrows():
        r = r.where(r.notna(), None).to_dict()
        assert interpret(r).primary == r["ground_truth"]


def test_rag_retrieves_preganglionic_principle_for_root_case():
    rec = {**_normal_case(), "R_ulnM_f": 34.5}
    _, hits = rag.retrieve_for_case(interpret(rec))
    assert any("pre-ganglionic" in c.section.lower() for c, _ in hits)


def test_critique_catches_hallucination_and_bad_physiology():
    rec = {**_normal_case(), "R_ulnS_amp": 8, "R_ulnM_cv": 42}
    truth = interpret(rec)
    fake = {"primary_pattern": "preganglionic_radiculopathy", "side": "R", "abnormal_parameters": ["R_ulnM_cv", "R_medM_amp"],
            "interpretation": "Ulnar CV 42 m/s and an F-wave of 36.8 ms [7]."}
    c = ai.critique(fake, rec, truth, None, n_context=4)
    assert not c.pattern_correct
    assert "36.8" in c.unsupported_numbers and "42" not in c.unsupported_numbers
    assert c.bad_citations == ["[7]"]
    assert "R_medM_amp" in c.invented and "R_ulnS_amp" in c.missed
    assert c.physiology_flags


def test_benchmark_with_mock_llm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(ai.llm, "complete", lambda *a, **k: json.dumps(
        {"primary_pattern": "normal", "side": "none", "abnormal_parameters": [], "interpretation": "Normal."}))
    cases = benchmark.select_cases(make_cohort(30, seed=2), 6)
    res = benchmark.run_llm(cases, "gemini", None)
    summary = benchmark.summarise(res)
    assert set(summary["condition"]) == {"rag", "no_rag"}
    assert isinstance(res, pd.DataFrame) and len(res) == 12
