"""Retrieval-augmented generation (RAG) over a curated electrophysiology knowledge base.

* Documents: Markdown files ``kb_*.md`` (one topic each), split into chunks at
  ``##`` headings so every retrieved passage is a self-contained, citable unit.
* Retriever: Okapi BM25 implemented from scratch (no external model download, fully
  inspectable, deterministic). Dense-embedding re-ranking is a natural next step.
* Query construction: the case's abnormal findings are translated into physiology terms
  (e.g. "prolonged F-wave ulnar, SNAP preserved") so retrieval is driven by the data,
  not only by the user's free-text question.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ncs_interpret import Interpretation
from ncs_schema import SIDES, STUDY_BY_KEY

KB_DIR = Path(__file__).resolve().parent
STOP = set("""a an the and or of to in on for with by is are be as at from that this it its
which than then into not no can may also when while where their there these those such
more most about above below over under per after before both each other""".split())


@dataclass
class Chunk:
    id: int
    doc: str
    section: str
    text: str

    @property
    def title(self) -> str:
        return f"{self.doc} - {self.section}"


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        # light suffix stripping so "latencies"~"latency", "prolonged"~"prolong"
        for suf in ("ies", "ed", "es", "s", "ing"):
            if len(t) > 4 and t.endswith(suf):
                t = t[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(t)
    return out


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("kb_*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = re.search(r"^# (.+)$", text, re.M)
        doc = doc_title.group(1).strip() if doc_title else path.stem
        for sec in re.split(r"^## ", text, flags=re.M)[1:]:
            head, _, body = sec.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(len(chunks), doc, head.strip(), body))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        sc = self.scores(query)
        ranked = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(self.chunks[i], sc[i]) for i in ranked if sc[i] > 0]


_INDEX: BM25 | None = None


def index() -> BM25:
    global _INDEX
    if _INDEX is None:
        _INDEX = BM25(load_chunks())
    return _INDEX


PARAM_TERMS = {
    "lat": "prolonged latency demyelination focal slowing",
    "amp": "reduced amplitude axonal loss",
    "cv": "slowed conduction velocity demyelination",
    "f": "prolonged F-wave proximal lesion root radiculopathy",
}


def case_query(interp: Interpretation, question: str = "") -> str:
    """Turn the structured abnormal findings into a retrieval query."""
    parts = [question, "cervical radiculopathy neck pain"]
    if not interp.abnormal:
        parts.append("normal nerve conduction does not exclude radiculopathy EMG")
    for a in interp.abnormal:
        s = STUDY_BY_KEY[a.study]
        kind = "SNAP sensory" if s.kind == "sensory" else "CMAP motor"
        parts.append(f"{s.nerve} {kind} {PARAM_TERMS[a.param]}")
    sens_abn = any(STUDY_BY_KEY[a.study].kind == "sensory" and a.param == "amp" for a in interp.abnormal)
    mot_abn = any(STUDY_BY_KEY[a.study].kind == "motor" for a in interp.abnormal)
    if mot_abn and not sens_abn:
        parts.append("SNAP preserved pre-ganglionic dorsal root ganglion")
    sides = {a.side for a in interp.abnormal}
    if len(sides) == 2:
        parts.append("bilateral multiple nerves polyneuropathy")
    return " ".join(p for p in parts if p)


def retrieve_for_case(interp: Interpretation, question: str = "", k: int = 4):
    q = case_query(interp, question)
    return q, index().search(q, k=k)


def format_context(hits) -> str:
    return "\n\n".join(f"[{i + 1}] ({c.title})\n{c.text}" for i, (c, _) in enumerate(hits))


def case_to_text(rec, interp: Interpretation, normals) -> str:
    """Compact, model-readable table of the measured values with reference limits."""
    from ncs_scoring import to_num
    lines = []
    for sd, side in SIDES.items():
        for s in STUDY_BY_KEY.values():
            vals = []
            for p in s.params:
                v = to_num(rec.get(f"{sd}_{s.key}_{p.key}"))
                if v is None:
                    continue
                op = "<=" if p.direction == "max" else ">="
                vals.append(f"{p.key}={v:g}{p.unit} (normal {op}{normals[f'{s.key}_{p.key}']:g})")
            if vals:
                lines.append(f"{sd}_{s.key} [{side} {s.name}]: " + "; ".join(vals))
    meta = []
    for k in ("age", "sex", "height_cm", "pain_side", "pain_duration_months", "paraesthesia", "limb_temp_c"):
        if rec.get(k) not in (None, ""):
            meta.append(f"{k}={rec.get(k)}")
    return "Patient: " + ", ".join(meta) + "\n" + "\n".join(lines)
