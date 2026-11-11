"""Deterministic, rule-based NCS interpretation.

This engine is (1) the reporting logic of the app and (2) the transparent reference
against which LLM interpretations are evaluated in the benchmark. Every rule is
written so a physiologist can audit it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ncs_schema import DEFAULT_NORMALS, SIDES, STUDIES, STUDY_BY_KEY
from ncs_scoring import to_num

# Controlled vocabulary shared by rules engine, LLM prompts and the benchmark.
PATTERNS = {
    "normal": "Normal study",
    "median_neuropathy_wrist": "Median neuropathy at the wrist (carpal tunnel pattern)",
    "ulnar_neuropathy": "Ulnar neuropathy",
    "radial_neuropathy": "Radial neuropathy",
    "preganglionic_radiculopathy": "Pre-ganglionic (root) pattern - cervical radiculopathy",
    "postganglionic_lesion": "Post-ganglionic lesion - plexopathy / peripheral neuropathy",
    "generalized_polyneuropathy": "Generalised polyneuropathy",
}
# When several patterns co-exist, the headline diagnosis follows this priority.
PRIORITY = ["generalized_polyneuropathy", "preganglionic_radiculopathy", "postganglionic_lesion",
            "median_neuropathy_wrist", "ulnar_neuropathy", "radial_neuropathy", "normal"]


@dataclass
class Abnormal:
    side: str
    study: str
    param: str
    value: float
    limit: float

    @property
    def code(self) -> str:
        return f"{self.side}_{self.study}_{self.param}"

    def describe(self) -> str:
        s = STUDY_BY_KEY[self.study]
        p = next(x for x in s.params if x.key == self.param)
        word = {"amp": "reduced", "cv": "slowed"}.get(self.param, "prolonged")
        op = "<=" if p.direction == "max" else ">="
        return (f"{SIDES[self.side]} {s.name.lower()}: {word} {p.label.lower()} "
                f"({self.value:g} {p.unit}; normal {op}{self.limit:g})")


@dataclass
class Interpretation:
    abnormal: list[Abnormal] = field(default_factory=list)
    patterns: dict[str, list[str]] = field(default_factory=lambda: {"R": [], "L": []})
    notes: list[str] = field(default_factory=list)
    any_data: bool = False

    @property
    def abnormal_codes(self) -> set[str]:
        return {a.code for a in self.abnormal}

    @property
    def primary(self) -> str:
        found = {p for sd in self.patterns.values() for p in sd}
        if not self.any_data:
            return "no_data"
        for p in PRIORITY:
            if p in found:
                return p
        return "normal" if not self.abnormal else "nonspecific_abnormal"

    @property
    def primary_side(self) -> str:
        p = self.primary
        if p in ("normal", "generalized_polyneuropathy", "no_data", "nonspecific_abnormal"):
            return "bilateral" if p == "generalized_polyneuropathy" else "none"
        sides = [sd for sd, pats in self.patterns.items() if p in pats]
        return "bilateral" if len(sides) == 2 else (sides[0] if sides else "none")

    def impression(self) -> str:
        if not self.any_data:
            return "No nerve conduction data entered."
        if not self.abnormal:
            return ("Nerve conduction studies of the tested median, ulnar and radial nerves "
                    "(motor, sensory and F-waves where performed) are within normal limits.\n\n"
                    "Note: a normal NCS does not exclude cervical radiculopathy; needle EMG may "
                    "be considered if clinically indicated.")
        parts = ["ABNORMAL STUDY."]
        for sd, name in SIDES.items():
            pats = [PATTERNS[p] for p in self.patterns[sd] if p != "normal"]
            if pats:
                parts.append(f"{name} upper limb:\n" + "\n".join(f"- {p}." for p in pats))
        if self.notes:
            parts.append("\n".join(f"- {n}." for n in self.notes))
        if len(parts) == 1:
            parts.append("Abnormal parameters as tabulated, without a specific localising pattern.")
        parts.append("Kindly correlate clinically.")
        return "\n\n".join(parts)


def _value(rec: Mapping[str, Any], side: str, study: str, param: str) -> float | None:
    return to_num(rec.get(f"{side}_{study}_{param}"))


def is_abnormal(study: str, param: str, value: float | None, normals: Mapping[str, float]) -> bool | None:
    lim = to_num(normals.get(f"{study}_{param}"))
    if value is None or lim is None:
        return None
    p = next(x for x in STUDY_BY_KEY[study].params if x.key == param)
    return value > lim if p.direction == "max" else value < lim


def interpret(rec: Mapping[str, Any], normals: Mapping[str, float] | None = None) -> Interpretation:
    N = {**DEFAULT_NORMALS, **(normals or {})}
    out = Interpretation()
    nerves_abn: dict[str, set[str]] = {"R": set(), "L": set()}

    for sd in SIDES:
        def A(study, param):
            return bool(is_abnormal(study, param, _value(rec, sd, study, param), N))

        def tested(study):
            return any(_value(rec, sd, study, p.key) is not None for p in STUDY_BY_KEY[study].params)

        for s in STUDIES:
            if not tested(s.key):
                continue
            out.any_data = True
            for p in s.params:
                v = _value(rec, sd, s.key, p.key)
                if is_abnormal(s.key, p.key, v, N):
                    out.abnormal.append(Abnormal(sd, s.key, p.key, v, float(N[f"{s.key}_{p.key}"])))
                    nerves_abn[sd].add(s.nerve)

        pats = out.patterns[sd]
        snap_abn = any(A(k, "amp") for k in ("medS", "ulnS", "radS"))
        snap_tested = any(tested(k) for k in ("medS", "ulnS", "radS"))
        motor_abn = any(A(k, "amp") for k in ("medM", "ulnM", "radM")) or A("medM", "f") or A("ulnM", "f")

        # Median neuropathy at the wrist: focal median slowing with normal ulnar sensory study.
        median_focal = A("medS", "lat") or A("medS", "cv") or A("medM", "lat")
        ulnar_sens_ok = not (A("ulnS", "lat") or A("ulnS", "cv") or A("ulnS", "amp"))
        if median_focal and ulnar_sens_ok:
            pats.append("median_neuropathy_wrist")
        # Ulnar neuropathy: ulnar motor slowing, or ulnar SNAP loss with a normal median SNAP.
        if A("ulnM", "cv") or (A("ulnS", "amp") and tested("medS") and not A("medS", "amp")):
            pats.append("ulnar_neuropathy")
        if any(A("radM", p) for p in ("amp", "cv", "lat")) or A("radS", "amp") or A("radS", "lat"):
            pats.append("radial_neuropathy")
        # Root lesions are proximal to the dorsal root ganglion -> SNAPs are spared.
        if motor_abn and snap_tested and not snap_abn:
            pats.append("preganglionic_radiculopathy")
        if motor_abn and snap_abn:
            pats.append("postganglionic_lesion")

        for s in (x for x in STUDIES if x.kind == "motor"):
            amp, cv, lat = (_value(rec, sd, s.key, k) for k in ("amp", "cv", "lat"))
            lA, lC, lL = (to_num(N.get(f"{s.key}_{k}")) for k in ("amp", "cv", "lat"))
            demy = (cv is not None and lC and cv < 0.75 * lC) or (lat is not None and lL and lat > 1.3 * lL)
            if demy:
                out.notes.append(f"{SIDES[sd]} {s.name.lower()}: slowing in the demyelinating range")
            elif amp is not None and lA and amp < lA:
                out.notes.append(f"{SIDES[sd]} {s.name.lower()}: low amplitude with preserved velocity (axonal)")

    for s in STUDIES:
        r, l = _value(rec, "R", s.key, "amp"), _value(rec, "L", s.key, "amp")
        if r and l:
            ratio = min(r, l) / max(r, l)
            if ratio < 0.5:
                lower = "right" if r < l else "left"
                out.notes.append(f"{s.name} amplitude asymmetry: {lower} {round((1 - ratio) * 100)}% lower")

    if len(nerves_abn["R"] & nerves_abn["L"]) >= 2:
        for sd in SIDES:
            out.patterns[sd].insert(0, "generalized_polyneuropathy")
    return out
