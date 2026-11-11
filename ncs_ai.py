"""LLM interpretation of an NCS case (with or without RAG) and automatic critique.

The critique step checks an AI interpretation against the measured data — the core
skill of evaluating AI-generated scientific claims:
  * Pattern / side agreement with the transparent rule engine.
  * Abnormal-parameter precision and recall (did it miss or invent abnormalities?).
  * Numeric fidelity: every number quoted in the narrative must exist in the input data
    or the reference limits; others are flagged as possible hallucinations.
  * Citation validity: cited passage numbers must exist in the retrieved context.
  * Physiology consistency: e.g. calling a case "radiculopathy" while a SNAP is reduced
    contradicts the pre-ganglionic principle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import ncs_llm as llm
import ncs_rag as rag
from ncs_interpret import PATTERNS, Interpretation, interpret
from ncs_schema import DEFAULT_NORMALS, STUDY_BY_KEY
from ncs_scoring import to_num

SYSTEM = (
    "You are a consultant clinical neurophysiologist reporting upper-limb nerve conduction "
    "studies for a cross-sectional study of patients with cervical (neck) pain. Reason from "
    "physiology: myelin vs axon, pre- vs post-ganglionic localisation, focal vs generalised. "
    "Use ONLY the measured values given; never invent measurements. If reference passages are "
    "provided, ground your reasoning in them and cite them as [n]. Reply with JSON only."
)

SCHEMA_TEXT = f"""Return a JSON object with exactly these keys:
  "primary_pattern": one of {list(PATTERNS)},
  "side": one of ["R", "L", "bilateral", "none"],
  "abnormal_parameters": list of codes like "R_medS_lat" (side_study_param) for every value outside its reference limit,
  "interpretation": 3-6 sentences of report-style reasoning quoting the key values{{CITE}},
  "differential": list of up to 3 alternative explanations,
  "confidence": number 0-1."""


def build_prompt(rec, normals=None, use_rag: bool = True, question: str = "", k: int = 4):
    N = {**DEFAULT_NORMALS, **(normals or {})}
    interp = interpret(rec, N)
    case_text = rag.case_to_text(rec, interp, N)
    hits, query = [], ""
    if use_rag:
        query, hits = rag.retrieve_for_case(interp, question, k=k)
    parts = [f"CASE DATA\n{case_text}"]
    if hits:
        parts.append("REFERENCE PASSAGES\n" + rag.format_context(hits))
    if question:
        parts.append(f"CLINICIAN QUESTION\n{question}")
    cite = ", citing reference passages as [n]" if hits else ""
    parts.append(SCHEMA_TEXT.replace("{CITE}", cite))
    return "\n\n".join(parts), hits, query, interp


@dataclass
class Critique:
    pattern_correct: bool
    side_correct: bool
    precision: float
    recall: float
    missed: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    physiology_flags: list[str] = field(default_factory=list)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    @property
    def issues(self) -> int:
        return (len(self.missed) + len(self.invented) + len(self.unsupported_numbers)
                + len(self.bad_citations) + len(self.physiology_flags) + (not self.pattern_correct))


def _allowed_numbers(rec, normals) -> set[float]:
    nums = set()
    for k, v in rec.items():
        f = to_num(v) if not isinstance(v, bool) else None
        if f is not None:
            nums.add(round(f, 2))
    for v in normals.values():
        nums.add(round(float(v), 2))
    return nums


def critique(output: dict, rec, truth: Interpretation, normals, n_context: int,
             truth_pattern: str | None = None, truth_side: str | None = None) -> Critique:
    N = {**DEFAULT_NORMALS, **(normals or {})}
    pred_codes = {str(c).strip() for c in output.get("abnormal_parameters", []) or []}
    true_codes = truth.abnormal_codes
    tp = len(pred_codes & true_codes)
    precision = tp / len(pred_codes) if pred_codes else (1.0 if not true_codes else 0.0)
    recall = tp / len(true_codes) if true_codes else 1.0

    tp_pat = truth_pattern or truth.primary
    tp_side = truth_side or truth.primary_side
    pred_side = str(output.get("side", "")).strip()
    side_ok = pred_side == tp_side or (tp_side == "none" and pred_side in ("none", ""))

    text = str(output.get("interpretation", ""))
    allowed = _allowed_numbers(rec, N)
    unsupported = []
    for m in re.finditer(r"(?<![\w\[.])(\d+(?:\.\d+)?)(?![\w\]%.\d])", text):
        val = float(m.group(1))
        # ignore small integers used as counts, citation numbers, percentages like 75% and digit names
        if val in (0, 1, 2, 3, 4, 5, 75, 130, 50, 100) or val in allowed:
            continue
        if not any(abs(val - a) <= 0.051 for a in allowed):
            unsupported.append(m.group(1))
    cites = {int(c) for c in re.findall(r"\[(\d+)\]", text)}
    bad = [f"[{c}]" for c in sorted(cites) if c < 1 or c > n_context]

    flags = []
    pat = output.get("primary_pattern")
    snap_low = any(STUDY_BY_KEY[a.study].kind == "sensory" and a.param == "amp" for a in truth.abnormal)
    if pat == "preganglionic_radiculopathy" and snap_low:
        flags.append("Called radiculopathy although a SNAP amplitude is reduced (SNAPs are spared in pre-ganglionic lesions).")
    if pat == "normal" and true_codes:
        flags.append("Reported normal despite values outside reference limits.")
    if pat not in PATTERNS:
        flags.append(f"Pattern '{pat}' is outside the allowed vocabulary.")

    return Critique(pattern_correct=(pat == tp_pat), side_correct=side_ok, precision=precision, recall=recall,
                    missed=sorted(true_codes - pred_codes), invented=sorted(pred_codes - true_codes),
                    unsupported_numbers=unsupported, bad_citations=bad, physiology_flags=flags)


def rule_engine_output(rec, normals=None) -> dict:
    """The deterministic engine expressed in the same JSON shape as the LLM (baseline)."""
    it = interpret(rec, {**DEFAULT_NORMALS, **(normals or {})})
    return {"primary_pattern": it.primary if it.primary in PATTERNS else "normal",
            "side": it.primary_side, "abnormal_parameters": sorted(it.abnormal_codes),
            "interpretation": it.impression(), "differential": [], "confidence": 1.0}


def run_case(rec, provider: str, model: str | None = None, normals=None, use_rag: bool = True,
             question: str = "", k: int = 4) -> dict:
    prompt, hits, query, interp = build_prompt(rec, normals, use_rag, question, k)
    raw = llm.complete(prompt, SYSTEM, provider, model)
    out = llm.parse_json(raw)
    return {"output": out, "raw": raw, "prompt": prompt, "hits": hits, "query": query, "truth": interp}
