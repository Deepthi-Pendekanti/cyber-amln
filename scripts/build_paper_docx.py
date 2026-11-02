"""Build an IEEE-style .docx of the AMLN paper directly from PAPER_IEEE.md.

Parsing the Markdown source (rather than keeping a second hand-maintained copy
of the text) guarantees the .docx can never drift from the paper. The generator
reproduces an IEEE layout: A4, Times New Roman, a single-column title/author
block, then a two-column body; section headings in small caps numerals; Markdown
tables rendered as real Word tables; and the ``![..](paper_fig*.png)`` image
references embedded as figures with their following italic caption line.

Run:  python scripts/build_paper_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "PAPER_IEEE.md"
OUT = ROOT / "PAPER_IEEE.docx"
BODY = "Times New Roman"

AUTHORS = [
    ["Sai Deepthi Pendekanti", "Integrated M.Tech Software Engineering",
     "Vellore Institute of Technology", "Vellore, India",
     "pendekanti.sai2023@vitstudent.ac.in"],
    ["Gowsik V", "Integrated M.Tech Software Engineering",
     "Vellore Institute of Technology", "Vellore, India",
     "gowsik.v2023@vitstudent.ac.in"],
    ["Srinidhi", "Integrated M.Tech Software Engineering",
     "Vellore Institute of Technology", "Vellore, India",
     "srinidhi1566@gmail.com"],
]


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def add_run(p, text, size=10, bold=False, italic=False):
    r = p.add_run(text)
    r.font.name = BODY
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    rpr = r._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), BODY)
    rfonts.set(qn("w:hAnsi"), BODY)
    return r


def heading(doc, text, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    add_run(p, text, size=size, bold=True)
    return p


def body_para(doc, text, size=10, italic=False, justify=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_run(p, text, size=size, italic=italic)
    return p


def centered(doc, text, size=10, bold=False, italic=False, space_after=2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    add_run(p, text, size=size, bold=bold, italic=italic)
    return p


def add_figure(doc, path: Path, caption: str, width_in=3.2):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(path), width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.paragraph_format.space_after = Pt(6)
    add_run(cap, caption, size=8)


def add_table(doc, header, rows, title_lines=None, notes=None):
    for i, line in enumerate(title_lines or []):
        centered(doc, line, size=8, bold=(i == 0), space_after=0)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for c, name in zip(t.rows[0].cells, header):
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(c.paragraphs[0], name, size=7, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for c, val in zip(cells, row):
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_run(c.paragraphs[0], str(val), size=7)
    for n in (notes or []):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        add_run(p, n, size=7)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# --------------------------------------------------------------------------
# Markdown parsing
# --------------------------------------------------------------------------
_INLINE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`")


def strip_inline(text: str) -> str:
    """Flatten Markdown emphasis/code markers to plain text."""
    return _INLINE.sub(lambda m: next(g for g in m.groups() if g is not None), text)


def parse_sections(md: str):
    """Split the paper into the body sections we render, dropping metadata."""
    lines = md.splitlines()
    # Only the real paper content lies between the ABSTRACT and REFERENCES.
    return lines


def table_block(lines, i):
    """If a Markdown table starts at line i, return (header, rows, next_i)."""
    if i >= len(lines) or not lines[i].lstrip().startswith("|"):
        return None
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    if i + 1 >= len(lines) or not re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
        return None
    rows = []
    j = i + 2
    while j < len(lines) and lines[j].lstrip().startswith("|"):
        rows.append([strip_inline(c.strip())
                     for c in lines[j].strip().strip("|").split("|")])
        j += 1
    return header, rows, j


# --------------------------------------------------------------------------
def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    lines = md.splitlines()

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY)

    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = Cm(1.9)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(1.59)

    # ---- single-column title + author block ----
    title = ("Adaptive Memory Landscape Network for Bounded Memory "
             "Real Time Streaming Cyber Threat Detection")
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_after = Pt(10)
    add_run(tp, title, size=20)

    at = doc.add_table(rows=1, cols=len(AUTHORS))
    at.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for cell, info in zip(at.rows[0].cells, AUTHORS):
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_run(cell.paragraphs[0], info[0], size=11)
        for line, it in [(info[1], False), (info[2], False),
                         (info[3], False), (info[4], True)]:
            q = cell.add_paragraph()
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after = Pt(0)
            add_run(q, line, size=9, italic=it)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ---- switch to two columns for the body ----
    new = doc.add_section(WD_SECTION.CONTINUOUS)
    cols = new._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "360")

    # ---- walk the Markdown body ----
    section_map = {
        "ABSTRACT": None, "KEYWORDS": None,
        "I. INTRODUCTION": "I.  INTRODUCTION",
        "II. RELATED WORK / LITERATURE SURVEY": "II.  RELATED WORK",
        "III. PROPOSED METHODOLOGY / SYSTEM DESIGN": "III.  PROPOSED METHODOLOGY",
        "IV. IMPLEMENTATION": "IV.  IMPLEMENTATION",
        "V. RESULTS AND DISCUSSION": "V.  RESULTS AND DISCUSSION",
        "VI. CONCLUSION AND FUTURE SCOPE": "VI.  CONCLUSION AND FUTURE SCOPE",
        "ACKNOWLEDGMENT": "ACKNOWLEDGMENT",
        "REFERENCES": "REFERENCES",
    }
    render = False          # start rendering at ABSTRACT
    stop = False
    fig_caption_re = re.compile(r"^!\[.*\]\((paper_fig\d[^)]*\.png)\)")
    pending_table_title = []
    i = 0
    n = len(lines)
    while i < n and not stop:
        raw = lines[i]
        line = raw.strip()

        # H2 section markers
        if line.startswith("## "):
            name = line[3:].strip()
            if name == "ABSTRACT":
                render = True
                heading(doc, "Abstract", size=10)
                i += 1
                continue
            if name in ("TITLE", "AUTHORS AND AFFILIATIONS"):
                render = False
                i += 1
                continue
            if name == "REMAINING TO DO BEFORE SUBMISSION" or \
               name == "REPRODUCIBILITY OF THE REPORTED NUMBERS":
                stop = True          # internal notes, not part of the paper
                break
            if name in section_map:
                render = True
                if section_map[name]:
                    heading(doc, section_map[name])
                elif name == "KEYWORDS":
                    heading(doc, "Keywords", size=10)
                i += 1
                continue
            i += 1
            continue

        if not render:
            i += 1
            continue

        # H3 subsection (e.g., "### A. System Architecture")
        if line.startswith("### "):
            heading(doc, line[4:].strip(), size=10)
            i += 1
            continue

        # horizontal rules / blockquotes / blanks
        if line in ("---", "") or line.startswith(">"):
            i += 1
            continue

        # figure reference -> embed image, then consume its caption line
        m = fig_caption_re.match(line)
        if m:
            img = ROOT / m.group(1)
            caption = ""
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip().startswith("Fig."):
                caption = strip_inline(lines[j].strip())
                j += 1
            add_figure(doc, img, caption)
            i = j
            continue

        # standalone TABLE title lines ("TABLE II", then its caption line)
        if re.match(r"^TABLE [IVX]+$", line):
            pending_table_title = [line]
            if i + 1 < n and lines[i + 1].strip() and \
               not lines[i + 1].lstrip().startswith("|"):
                pending_table_title.append(lines[i + 1].strip())
                i += 2
            else:
                i += 1
            continue

        # equation lines (contain "(1)".."(4)") -> centered monospace-ish
        if re.search(r"\(\d\)\s*$", line) and ("=" in line or "<=" in line):
            centered(doc, line, size=10, space_after=4)
            i += 1
            continue

        # Markdown table
        tb = table_block(lines, i)
        if tb:
            header, rows, nxt = tb
            notes = []
            k = nxt
            while k < n and re.match(r"^[a-d]\.\s", lines[k].strip()):
                notes.append(strip_inline(lines[k].strip()))
                k += 1
            add_table(doc, header, rows, title_lines=pending_table_title,
                      notes=notes)
            pending_table_title = []
            i = k
            continue

        # numbered sub-point like "1) Surprise gated plasticity: ..."
        # or a reference "[1] ..." in the REFERENCES section -> normal paragraph.
        # default: a body paragraph (join wrapped lines until blank).
        para = [strip_inline(line)]
        j = i + 1
        while j < n and lines[j].strip() and not lines[j].strip().startswith("## ") \
                and not lines[j].strip().startswith("### ") \
                and not lines[j].strip().startswith("|") \
                and not lines[j].strip().startswith("![") \
                and lines[j].strip() != "---":
            para.append(strip_inline(lines[j].strip()))
            j += 1
        text = " ".join(para)
        is_abstract = text.lower().startswith("modern intrusion detection")
        body_para(doc, text, size=9 if is_abstract else 10,
                  italic=is_abstract)
        i = j

    doc.save(OUT)
    print(f"Saved {OUT}  (parsed directly from {SRC.name})")


if __name__ == "__main__":
    main()
