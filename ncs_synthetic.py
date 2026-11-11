"""Physiologically plausible synthetic cohort with known ground truth.

Used for (a) a public demo that contains NO real patient data and (b) an AI benchmark
where the correct electrodiagnostic pattern is known by construction.

Design choices (documented so they can be critiqued):
* Normal values are drawn from distributions centred well inside the reference limits.
* F-wave latency scales with height (longer limb -> longer F latency), ~0.1 ms per cm.
* Pathology is injected on the symptomatic side according to the pathophysiology:
  - Carpal tunnel: focal median slowing at the wrist (sensory latency/CV, +/- motor DL).
  - C8 radiculopathy (pre-ganglionic): low ulnar CMAP and/or prolonged F-waves, SNAPs spared
    because the dorsal root ganglion and its peripheral axon are intact.
  - Ulnar neuropathy: ulnar motor CV slowing and/or low ulnar SNAP, median normal.
  - Superficial radial neuropathy: low radial SNAP.
  - Polyneuropathy: bilateral, length-dependent sensory > motor loss with mild slowing.
* Each case is accepted only if the transparent rule engine reproduces the intended
  label, so the benchmark's ground truth is unambiguous.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ncs_interpret import interpret
from ncs_schema import NDI_COLS

LABEL_MIX = {
    "normal": 0.42,
    "preganglionic_radiculopathy": 0.22,
    "median_neuropathy_wrist": 0.20,
    "ulnar_neuropathy": 0.07,
    "generalized_polyneuropathy": 0.05,
    "radial_neuropathy": 0.04,
}

NORMAL_DIST = {  # (mean, sd, low clip, high clip)
    "medM_lat": (3.4, 0.30, 2.6, 4.1), "medM_amp": (9.0, 2.0, 5.0, 15), "medM_cv": (57, 3, 51, 66),
    "ulnM_lat": (2.6, 0.25, 2.0, 3.1), "ulnM_amp": (10.0, 1.8, 6.8, 15), "ulnM_cv": (60, 3, 52, 68),
    "radM_lat": (2.2, 0.25, 1.6, 2.7), "radM_amp": (5.0, 1.2, 2.8, 9), "radM_cv": (61, 3.5, 52, 70),
    "medS_lat": (2.9, 0.20, 2.4, 3.3), "medS_amp": (40, 10, 24, 70), "medS_cv": (57, 3.5, 51, 66),
    "ulnS_lat": (2.6, 0.20, 2.1, 3.0), "ulnS_amp": (32, 8, 20, 55), "ulnS_cv": (57, 3.5, 51, 66),
    "radS_lat": (2.2, 0.20, 1.8, 2.7), "radS_amp": (28, 7, 17, 50), "radS_cv": (57, 3.5, 51, 66),
}


def _normal_limb(rng: np.random.Generator, height: float) -> dict[str, float]:
    v = {k: float(np.clip(rng.normal(m, s), lo, hi)) for k, (m, s, lo, hi) in NORMAL_DIST.items()}
    v["medM_f"] = float(np.clip(26.0 + 0.10 * (height - 165) + rng.normal(0, 1.0), 22, 30.5))
    v["ulnM_f"] = float(np.clip(26.5 + 0.10 * (height - 165) + rng.normal(0, 1.0), 22, 31.5))
    return v


def _inject(rng, limb: dict, label: str) -> None:
    u = rng.uniform
    if label == "median_neuropathy_wrist":
        limb["medS_lat"] = u(3.7, 4.8)
        limb["medS_cv"] = u(35, 47)
        if rng.random() < 0.6:
            limb["medM_lat"] = u(4.6, 6.2)
        if rng.random() < 0.3:
            limb["medS_amp"] = u(9, 19)
    elif label == "preganglionic_radiculopathy":
        limb["ulnM_f"] = u(32.6, 36)
        if rng.random() < 0.6:
            limb["ulnM_amp"] = u(2.5, 5.6)
        if rng.random() < 0.4:
            limb["medM_f"] = u(31.6, 34.5)
    elif label == "ulnar_neuropathy":
        limb["ulnM_cv"] = u(38, 47)
        if rng.random() < 0.7:
            limb["ulnS_amp"] = u(6, 15)
    elif label == "radial_neuropathy":
        limb["radS_amp"] = u(4, 13)
    elif label == "generalized_polyneuropathy":
        limb["medS_amp"] = u(7, 18)
        limb["ulnS_amp"] = u(5, 15)
        limb["medM_cv"] = u(41, 47.5)
        limb["ulnM_cv"] = u(41, 47.5)


def _one_case(rng, idx: int, label: str) -> dict:
    sex = rng.choice(["Male", "Female"])
    age = int(np.clip(rng.normal(42, 11), 20, 70))
    height = float(np.clip(rng.normal(168 if sex == "Male" else 155, 6.5), 140, 190))
    bmi = float(np.clip(rng.normal(24.5, 3.5), 17, 36))
    weight = bmi * (height / 100) ** 2
    pain_side = rng.choice(["Right", "Left", "Bilateral"], p=[0.45, 0.4, 0.15])
    affected = {"Right": ["R"], "Left": ["L"], "Bilateral": ["R", "L"]}[pain_side]
    if label not in ("normal", "generalized_polyneuropathy"):
        affected = [affected[0]] if rng.random() < 0.85 else ["R", "L"]
    duration = float(np.round(rng.gamma(2.0, 6.0 if label != "normal" else 4.0), 1))
    vas = float(np.clip(rng.normal(6.0 if label != "normal" else 4.5, 1.5), 1, 10))

    rec: dict = {
        "study_id": f"SYN-{idx:03d}", "age": age, "sex": sex, "date": "",
        "height_cm": round(height, 1), "weight_kg": round(weight, 1),
        "occupation": rng.choice(["Desk/IT", "Teacher", "Homemaker", "Driver", "Manual labour", "Healthcare"]),
        "pain_side": pain_side, "pain_duration_months": duration, "vas": round(vas, 1),
        "paraesthesia": "None" if label == "normal" and rng.random() < 0.6 else pain_side,
        "limb_temp_c": round(float(rng.normal(32.5, 0.6)), 1),
        "ground_truth": label,
        "ground_truth_side": ("none" if label == "normal" else
                              "bilateral" if label == "generalized_polyneuropathy" or len(affected) == 2
                              else affected[0]),
    }
    # NDI items track pain intensity (VAS) with item-level noise.
    for c in NDI_COLS:
        rec[c] = int(np.clip(np.round(rng.normal(vas / 2.6, 0.9)), 0, 5))
    if rng.random() < 0.15:
        rec["ndi_q8"] = None  # Driving: often not applicable
    # IPAQ: higher activity loosely associated with lower disability.
    rec.update({
        "ipaq_vig_days": int(rng.choice([0, 0, 0, 1, 2, 3, 4])), "ipaq_vig_min": int(rng.choice([0, 20, 30, 45, 60])),
        "ipaq_mod_days": int(rng.integers(0, 6)), "ipaq_mod_min": int(rng.choice([0, 15, 20, 30, 45, 60])),
        "ipaq_walk_days": int(rng.integers(1, 8)), "ipaq_walk_min": int(rng.choice([10, 15, 20, 30, 45, 60])),
        "ipaq_sit_min": int(np.clip(rng.normal(420, 120), 120, 720)),
    })
    # Body composition: Deurenberg-style body-fat estimate + noise.
    fat = 1.2 * bmi + 0.23 * age - 10.8 * (sex == "Male") - 5.4 + rng.normal(0, 2.5)
    ffm = weight * (1 - fat / 100)
    rec.update({
        "bc_fat_pct": round(float(fat), 1), "bc_ffm_kg": round(float(ffm), 1),
        "bc_muscle_kg": round(float(ffm * 0.53 + rng.normal(0, 1)), 1),
        "bc_visceral": int(np.clip(round((bmi - 18) * 0.9 + (age - 30) * 0.08 + rng.normal(0, 1.5)), 1, 20)),
        "bc_tbw_l": round(float(ffm * 0.73), 1), "bc_bmr_kcal": int(370 + 21.6 * ffm),
    })
    for sd in ("R", "L"):
        limb = _normal_limb(rng, height)
        if label == "generalized_polyneuropathy" or (label != "normal" and sd in affected):
            _inject(rng, limb, label)
        for k, v in limb.items():
            rec[f"{sd}_{k}"] = round(v, 1) if not k.endswith("_lat") else round(v, 2)
    if rng.random() < 0.05:  # radial sensory occasionally not performed
        for sd in ("R", "L"):
            for p in ("lat", "amp", "cv"):
                rec[f"{sd}_radS_{p}"] = None
    return rec


def make_cohort(n: int = 150, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(LABEL_MIX), size=n, p=list(LABEL_MIX.values()))
    rows = []
    for i, lab in enumerate(labels, start=1):
        for _ in range(200):  # rejection sampling: keep only unambiguous cases
            rec = _one_case(rng, i, lab)
            res = interpret(rec)
            if res.primary == lab and (lab == "normal" or res.primary_side == rec["ground_truth_side"]):
                rows.append(rec)
                break
    return pd.DataFrame(rows)


if __name__ == "__main__":  # python ncs_synthetic.py
    df = make_cohort()
    df.to_csv("demo_cohort.csv", index=False)
    print(df["ground_truth"].value_counts())
