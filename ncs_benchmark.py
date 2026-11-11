"""Benchmark: how well does an LLM interpret nerve conduction studies?

Cases come from the synthetic cohort, where the electrodiagnostic pattern is known by
construction. Each model run is scored with ``ai.critique``. The same harness scores
two reference systems so model results have context:

* ``rule_engine`` - transparent deterministic rules (upper bound on this synthetic set).
* ``always_normal`` - trivial baseline (lower bound; equals the prevalence of normals).

Conditions: ``rag`` (retrieved guideline passages in the prompt) vs ``no_rag``, an
ablation that isolates what retrieval adds.
"""
from __future__ import annotations

import pandas as pd

import ncs_ai as ai
from ncs_interpret import interpret
from ncs_llm import LLMError


def _score_row(case, out, n_ctx, system, condition, error=""):
    truth = interpret(case)
    c = ai.critique(out, case, truth, None, n_ctx, case.get("ground_truth"), case.get("ground_truth_side"))
    return {
        "study_id": case["study_id"], "system": system, "condition": condition,
        "ground_truth": case.get("ground_truth"), "predicted": out.get("primary_pattern"),
        "pattern_correct": c.pattern_correct, "side_correct": c.side_correct,
        "param_precision": c.precision, "param_recall": c.recall, "param_f1": c.f1,
        "missed": ", ".join(c.missed), "invented": ", ".join(c.invented),
        "unsupported_numbers": ", ".join(c.unsupported_numbers), "bad_citations": ", ".join(c.bad_citations),
        "physiology_flags": " | ".join(c.physiology_flags), "issues": c.issues, "error": error,
        "interpretation": out.get("interpretation", ""),
    }


def select_cases(cohort: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified sample so every pattern is represented."""
    per = max(1, n // cohort["ground_truth"].nunique())
    sample = pd.concat([g.sample(min(len(g), per), random_state=seed)
                        for _, g in cohort.groupby("ground_truth")])
    if len(sample) < n:
        rest = cohort.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(len(rest), n - len(sample)), random_state=seed)])
    return sample.head(n)


def run_baselines(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, case in cases.iterrows():
        case = case.where(case.notna(), None).to_dict()
        rows.append(_score_row(case, ai.rule_engine_output(case), 0, "rule_engine", "-"))
        rows.append(_score_row(case, {"primary_pattern": "normal", "side": "none", "abnormal_parameters": [],
                                      "interpretation": ""}, 0, "always_normal", "-"))
    return pd.DataFrame(rows)


def run_llm(cases: pd.DataFrame, provider: str, model: str | None, conditions=("rag", "no_rag"),
            progress=None) -> pd.DataFrame:
    rows, total, done = [], len(cases) * len(conditions), 0
    label = f"{provider}:{model or ai.llm.DEFAULT_MODELS[provider]}"
    for _, case in cases.iterrows():
        case = case.where(case.notna(), None).to_dict()
        for cond in conditions:
            try:
                res = ai.run_case(case, provider, model, use_rag=(cond == "rag"))
                rows.append(_score_row(case, res["output"], len(res["hits"]), label, cond))
            except (LLMError, ValueError) as e:
                rows.append({"study_id": case["study_id"], "system": label, "condition": cond,
                             "ground_truth": case.get("ground_truth"), "error": str(e)[:200]})
            done += 1
            if progress:
                progress(done / total)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["error"].fillna("") == ""] if "error" in results.columns else results
    g = ok.groupby(["system", "condition"])
    return pd.DataFrame({
        "n": g.size(),
        "pattern_accuracy": g["pattern_correct"].mean(),
        "side_accuracy": g["side_correct"].mean(),
        "param_F1": g["param_f1"].mean(),
        "cases_with_unsupported_numbers": g["unsupported_numbers"].apply(lambda s: (s.fillna("") != "").mean()),
        "physiology_errors": g["physiology_flags"].apply(lambda s: (s.fillna("") != "").mean()),
        "mean_issues": g["issues"].mean(),
    }).reset_index()
