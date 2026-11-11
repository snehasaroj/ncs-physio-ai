"""Questionnaire and anthropometric scoring: BMI, Neck Disability Index, IPAQ-short."""
from __future__ import annotations

import math
from typing import Any, Mapping

from ncs_schema import NDI_COLS


def to_num(v: Any) -> float | None:
    """Convert form/CSV values to float; blanks, None and NaN become None."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ---------------- BMI ----------------
def bmi(height_cm: Any, weight_kg: Any) -> float | None:
    h, w = to_num(height_cm), to_num(weight_kg)
    if not h or not w:
        return None
    return w / (h / 100) ** 2


def bmi_category(b: float | None) -> str:
    """WHO Asia-Pacific cut-offs (appropriate for an Indian cohort)."""
    if b is None:
        return ""
    if b < 18.5:
        return "Underweight"
    if b < 23:
        return "Normal"
    if b < 25:
        return "Overweight"
    return "Obese"


# ---------------- NDI ----------------
def ndi(rec: Mapping[str, Any]) -> dict:
    """Neck Disability Index (Vernon & Mior). 10 sections scored 0-5.

    Unanswered sections are excluded and the percentage is computed over answered
    sections (score / (answered*5) * 100). Severity uses the /50 scale bands.
    """
    vals = [to_num(rec.get(c)) for c in NDI_COLS]
    ans = [v for v in vals if v is not None]
    if not ans:
        return {"total": None, "answered": 0, "percent": None, "severity": ""}
    total = sum(ans)
    pct = total / (len(ans) * 5) * 100
    s50 = pct / 2
    if s50 < 5:
        sev = "No disability"
    elif s50 < 15:
        sev = "Mild"
    elif s50 < 25:
        sev = "Moderate"
    elif s50 < 35:
        sev = "Severe"
    else:
        sev = "Complete"
    return {"total": total, "answered": len(ans), "percent": pct, "severity": sev}


# ---------------- IPAQ short form ----------------
def ipaq(rec: Mapping[str, Any]) -> dict:
    """IPAQ short-form scoring following the official scoring protocol.

    MET values: walking 3.3, moderate 4.0, vigorous 8.0. Bouts < 10 min are set to 0
    and daily minutes are truncated at 180 (data-cleaning rules).
    """
    def days(k):
        return min(7.0, max(0.0, to_num(rec.get(k)) or 0.0))

    def mins(k):
        v = to_num(rec.get(k)) or 0.0
        return 0.0 if v < 10 else min(180.0, v)

    if all(to_num(rec.get(k)) is None for k in ("ipaq_vig_days", "ipaq_mod_days", "ipaq_walk_days")):
        return {"walk_met": None, "mod_met": None, "vig_met": None, "total_met": None, "category": ""}

    vd, md, wd = days("ipaq_vig_days"), days("ipaq_mod_days"), days("ipaq_walk_days")
    vm = mins("ipaq_vig_min") if vd else 0.0
    mm = mins("ipaq_mod_min") if md else 0.0
    wm = mins("ipaq_walk_min") if wd else 0.0
    walk, mod, vig = 3.3 * wm * wd, 4.0 * mm * md, 8.0 * vm * vd
    total = walk + mod + vig
    active_days = (vd if vm else 0) + (md if mm else 0) + (wd if wm else 0)

    if (vd >= 3 and vig >= 1500) or (active_days >= 7 and total >= 3000):
        cat = "High"
    elif (vd >= 3 and vm >= 20) or ((md if mm >= 30 else 0) + (wd if wm >= 30 else 0) >= 5) \
            or (active_days >= 5 and total >= 600):
        cat = "Moderate"
    else:
        cat = "Low"
    return {"walk_met": walk, "mod_met": mod, "vig_met": vig, "total_met": total, "category": cat}


def derived(rec: Mapping[str, Any]) -> dict:
    """All derived (computed) variables for one record, flat, analysis-ready."""
    b = bmi(rec.get("height_cm"), rec.get("weight_kg"))
    n = ndi(rec)
    i = ipaq(rec)
    return {
        "bmi": b, "bmi_category": bmi_category(b),
        "ndi_total": n["total"], "ndi_answered": n["answered"],
        "ndi_percent": n["percent"], "ndi_severity": n["severity"],
        "ipaq_walk_met": i["walk_met"], "ipaq_mod_met": i["mod_met"],
        "ipaq_vig_met": i["vig_met"], "ipaq_total_met": i["total_met"],
        "ipaq_category": i["category"],
    }
