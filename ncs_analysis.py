"""Cohort-level analysis: analysis-ready table, descriptives and hypothesis tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ncs_interpret import interpret
from ncs_schema import DEFAULT_NORMALS, SIDES, STUDIES
from ncs_scoring import derived


def analysis_table(df: pd.DataFrame, normals=None) -> pd.DataFrame:
    """Inputs + derived scores + per-parameter abnormal flags + symptomatic-side values."""
    N = {**DEFAULT_NORMALS, **(normals or {})}
    rows = []
    for _, r in df.iterrows():
        rec = r.where(r.notna(), None).to_dict()
        it = interpret(rec, N)
        d = derived(rec)
        d["ncs_result"] = "No data" if not it.any_data else ("Abnormal" if it.abnormal else "Normal")
        d["ncs_pattern"] = it.primary
        d["n_abnormal_params"] = len(it.abnormal)
        codes = it.abnormal_codes
        for sd in SIDES:
            for s in STUDIES:
                for p in s.params:
                    c = f"{sd}_{s.key}_{p.key}"
                    d[c + "_abn"] = int(c in codes) if rec.get(c) is not None else None
        # Symptomatic side: the painful side (worse side = right if bilateral/central)
        side = {"Left": "L"}.get(rec.get("pain_side"), "R")
        for s in STUDIES:
            for p in s.params:
                d[f"sym_{s.key}_{p.key}"] = rec.get(f"{side}_{s.key}_{p.key}")
        rows.append(d)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def correlations(t: pd.DataFrame, x_cols: list[str], y_cols: list[str]) -> pd.DataFrame:
    """Spearman correlations with BH-FDR correction across all tests."""
    rows = []
    for x in x_cols:
        for y in y_cols:
            sub = t[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 8 or sub[x].nunique() < 3 or sub[y].nunique() < 3:
                continue
            rho, p = stats.spearmanr(sub[x], sub[y])
            rows.append({"x": x, "y": y, "n": len(sub), "rho": rho, "p": p})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_fdr"] = bh_fdr(res["p"].values)
        res = res.sort_values("p")
    return res


def compare_groups(t: pd.DataFrame, group: str, value_cols: list[str]) -> pd.DataFrame:
    """Median (IQR) by group; Mann-Whitney U for 2 groups, Kruskal-Wallis for >2."""
    rows = []
    groups = [g for g in t[group].dropna().unique() if g != ""]
    for v in value_cols:
        data = {g: pd.to_numeric(t.loc[t[group] == g, v], errors="coerce").dropna() for g in groups}
        data = {g: s for g, s in data.items() if len(s) >= 3}
        if len(data) < 2:
            continue
        row = {"variable": v}
        for g, s in data.items():
            row[str(g)] = f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f}), n={len(s)}"
        if len(data) == 2:
            a, b = data.values()
            row["test"], row["p"] = "Mann-Whitney U", stats.mannwhitneyu(a, b).pvalue
        else:
            row["test"], row["p"] = "Kruskal-Wallis", stats.kruskal(*data.values()).pvalue
        rows.append(row)
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_fdr"] = bh_fdr(res["p"].values)
    return res


def chi_square(t: pd.DataFrame, a: str, b: str):
    ct = pd.crosstab(t[a], t[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, None
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return ct, {"chi2": chi2, "dof": dof, "p": p}
