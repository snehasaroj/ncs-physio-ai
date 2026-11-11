"""A4 PDF report for one patient (reportlab)."""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ncs_interpret import interpret, is_abnormal
from ncs_schema import DEFAULT_NORMALS, F_WAVE, MOTOR_PARAMS, SENSORY_PARAMS, SIDES, STUDIES
from ncs_scoring import derived, to_num

NAVY = colors.HexColor("#0d2b3a")
RED = colors.HexColor("#b42318")


def _fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def build_pdf(rec: dict, settings: dict | None = None, normals=None, impression: str | None = None) -> bytes:
    S = {"institution": "Rajiv Gandhi Medical College, Thane",
         "department": "Department of Physiology - Neurophysiology Laboratory",
         "verified_by": "", **(settings or {})}
    N = {**DEFAULT_NORMALS, **(normals or {})}
    it = interpret(rec, N)
    d = derived(rec)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title=f"NCS report {rec.get('study_id', '')}")
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontSize=14, textColor=NAVY, spaceAfter=0)
    sub = ParagraphStyle("s", parent=ss["Normal"], alignment=1, fontSize=9)
    sec = ParagraphStyle("sec", parent=ss["Heading4"], textColor=NAVY, spaceBefore=6, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.5, leading=11)
    grid = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]

    el = [Paragraph(S["institution"], h), Paragraph(S["department"], sub),
          Paragraph("<b>NERVE CONDUCTION STUDY REPORT</b>", sub), Spacer(1, 6)]

    g = lambda k: "-" if rec.get(k) in (None, "") else str(rec.get(k))
    info = [["Study ID", g("study_id"), "Age / Sex", f"{g('age')} / {g('sex')}", "Date", g("date")],
            ["Height / Weight", f"{g('height_cm')} cm / {g('weight_kg')} kg", "BMI",
             f"{_fmt(d['bmi'])} ({d['bmi_category']})" if d["bmi"] else "-", "Limb temp.", g("limb_temp_c") + " C"],
            ["Neck pain side", g("pain_side"), "Duration", g("pain_duration_months") + " mo", "VAS", g("vas")],
            ["NDI", f"{_fmt(d['ndi_percent'], 0)}% ({d['ndi_severity']})" if d["ndi_percent"] is not None else "-",
             "IPAQ", f"{_fmt(d['ipaq_total_met'], 0)} MET-min/wk ({d['ipaq_category']})" if d["ipaq_total_met"] is not None else "-",
             "Body fat", (g("bc_fat_pct") + " %") if rec.get("bc_fat_pct") is not None else "-"]]
    t = Table(info, colWidths=[24 * mm, 36 * mm, 20 * mm, 44 * mm, 20 * mm, 38 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                           ("TEXTCOLOR", (2, 0), (2, -1), colors.grey), ("TEXTCOLOR", (4, 0), (4, -1), colors.grey)]))
    el.append(t)

    def ncs_table(kind, params, title):
        studies = [s for s in STUDIES if s.kind == kind]
        rows = [["Nerve", "Side"] + [f"{p.label} ({p.unit})" for p in params]]
        styles = list(grid)
        for s in studies:
            for sd, side in SIDES.items():
                vals = [to_num(rec.get(f"{sd}_{s.key}_{p.key}")) for p in params]
                if all(v is None for v in vals):
                    continue
                row = [s.nerve, side]
                for j, (p, v) in enumerate(zip(params, vals)):
                    ab = is_abnormal(s.key, p.key, v, N)
                    row.append("-" if v is None else f"{v:g}{' *' if ab else ''}")
                    if ab:
                        styles.append(("TEXTCOLOR", (2 + j, len(rows)), (2 + j, len(rows)), RED))
                rows.append(row)
        if len(rows) == 1:
            return []
        lim = ["Normal", ""] + [" / ".join(f"{'<=' if p.direction == 'max' else '>='}{N[f'{s.key}_{p.key}']:g}"
                                           for s in studies if f"{s.key}_{p.key}" in N) for p in params]
        rows.append(lim)
        styles.append(("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "Helvetica-Oblique"))
        w = (182 - 38) / len(params)
        tb = Table(rows, repeatRows=1, colWidths=[22 * mm, 16 * mm] + [w * mm] * len(params))
        tb.setStyle(TableStyle(styles))
        return [Paragraph(title, sec), tb]

    el += ncs_table("motor", list(MOTOR_PARAMS), "Motor nerve conduction")
    frows = [["Nerve", "Right (ms)", "Left (ms)", "Normal (ms)"]]
    fst = list(grid)
    for s in (x for x in STUDIES if x.has_f_wave):
        r, l = to_num(rec.get(f"R_{s.key}_f")), to_num(rec.get(f"L_{s.key}_f"))
        if r is None and l is None:
            continue
        row = [s.nerve]
        for j, v in enumerate((r, l)):
            ab = is_abnormal(s.key, "f", v, N)
            row.append("-" if v is None else f"{v:g}{' *' if ab else ''}")
            if ab:
                fst.append(("TEXTCOLOR", (1 + j, len(frows)), (1 + j, len(frows)), RED))
        row.append(f"<={N[f'{s.key}_f']:g}")
        frows.append(row)
    if len(frows) > 1:
        tb = Table(frows, repeatRows=1, colWidths=[38 * mm, 48 * mm, 48 * mm, 48 * mm])
        tb.setStyle(TableStyle(fst))
        el += [Paragraph("F-wave study", sec), tb]
    el += ncs_table("sensory", list(SENSORY_PARAMS), "Sensory nerve conduction")
    el.append(Paragraph("* outside reference limits", body))
    el.append(Paragraph("Impression", sec))
    imp = (impression or it.impression()).replace("\n", "<br/>")
    box = Table([[Paragraph(imp, body)]], colWidths=[182 * mm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    el += [box, Spacer(1, 30)]
    sig = Table([["____________________", "____________________"], ["Examiner", S.get("verified_by") or "Verified by"]],
                colWidths=[91 * mm, 91 * mm])
    sig.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    el += [sig, Spacer(1, 8), Paragraph(
        f"Generated {datetime.now():%d %b %Y %H:%M}. Interpretation is computer-assisted and must be verified "
        "by the reporting physician.", ParagraphStyle("f", parent=body, fontSize=7, textColor=colors.grey))]
    doc.build(el)
    return buf.getvalue()
