#!/usr/bin/env python3
"""Render the beginner RCBF_SAC Markdown tutorial as a navigable PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "RCBF_SAC_5.2_逐行讲解.md"
OUTPUT = ROOT / "RCBF_SAC_5.2_逐行讲解.pdf"


class TutorialDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(
            PageTemplate(id="tutorial", frames=frame, onPage=self._draw_page)
        )

    def _draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("STSong-Light", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            doc.leftMargin,
            12 * mm,
            "RCBF_SAC 5.2 初学者逐行讲解",
        )
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            12 * mm,
            f"第 {doc.page} 页",
        )
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Heading1", "Heading2", "Heading3"}:
            return

        level = {"Heading1": 0, "Heading2": 1, "Heading3": 2}[flowable.style.name]
        text = flowable.getPlainText()
        key = getattr(flowable, "_bookmark_name", None)
        if key is None:
            return

        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def make_styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    styles = getSampleStyleSheet()
    base = {
        "fontName": "STSong-Light",
        "textColor": colors.HexColor("#202124"),
        "leading": 17,
        "fontSize": 10.5,
        "spaceAfter": 5,
        "wordWrap": "CJK",
    }

    styles.add(
        ParagraphStyle(
            name="ChineseBody",
            parent=styles["BodyText"],
            **base,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseBullet",
            parent=styles["BodyText"],
            **base,
            leftIndent=14,
            firstLineIndent=0,
            bulletIndent=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChineseQuote",
            parent=styles["BodyText"],
            **base,
            leftIndent=12,
            rightIndent=8,
            borderColor=colors.HexColor("#8aa4c8"),
            borderWidth=1,
            borderPadding=6,
            backColor=colors.HexColor("#f3f6fa"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="STSong-Light",
            fontSize=8.2,
            leading=11,
            leftIndent=7,
            rightIndent=7,
            borderColor=colors.HexColor("#d7dce2"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#f7f8fa"),
            textColor=colors.HexColor("#1f2933"),
            spaceBefore=3,
            spaceAfter=8,
            splitLongWords=False,
        )
    )

    styles["Title"].fontName = "STSong-Light"
    styles["Title"].fontSize = 24
    styles["Title"].leading = 32
    styles["Title"].alignment = TA_CENTER
    styles["Title"].textColor = colors.HexColor("#183b66")
    styles["Title"].spaceAfter = 16

    for name, size, leading, color in (
        ("Heading1", 17, 24, "#163a63"),
        ("Heading2", 14, 20, "#24527a"),
        ("Heading3", 12, 18, "#365f7d"),
    ):
        styles[name].fontName = "STSong-Light"
        styles[name].fontSize = size
        styles[name].leading = leading
        styles[name].textColor = colors.HexColor(color)
        styles[name].spaceBefore = 12
        styles[name].spaceAfter = 7
        styles[name].keepWithNext = True

    return styles


TOKEN_RE = re.compile(r"(`[^`]+`|\[[^\]]+\]\([^)]+\))")


def inline_markup(text: str) -> str:
    parts = TOKEN_RE.split(text)
    rendered = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            font_name = (
                "Courier"
                if all(ord(character) < 128 for character in part[1:-1])
                else "STSong-Light"
            )
            rendered.append(
                f'<font name="{font_name}" color="#7a2e2e">{html.escape(part[1:-1])}</font>'
            )
            continue
        if part.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
            if match:
                label, target = match.groups()
                rendered.append(
                    '<link href="{}" color="#175ea8"><u>{}</u></link>'.format(
                        html.escape(target, quote=True),
                        html.escape(label),
                    )
                )
                continue
        rendered.append(html.escape(part))
    return "".join(rendered)


def add_heading(story, styles, level: int, text: str, counter: int):
    style_name = "Title" if level == 0 else f"Heading{min(level, 3)}"
    paragraph = Paragraph(inline_markup(text), styles[style_name])
    if level > 0:
        paragraph._bookmark_name = f"heading-{counter}"
    story.append(paragraph)


def parse_markdown(text: str, styles):
    story = []
    lines = text.splitlines()
    paragraph_lines = []
    code_lines = []
    in_code = False
    heading_counter = 0
    title_seen = False

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            joined = " ".join(line.strip() for line in paragraph_lines)
            story.append(Paragraph(inline_markup(joined), styles["ChineseBody"]))
            paragraph_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(
                    Preformatted(
                        "\n".join(code_lines),
                        styles["CodeBlock"],
                        maxLineLength=105,
                    )
                )
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text_value = heading.group(2)
            if not title_seen:
                add_heading(story, styles, 0, text_value, heading_counter)
                story.append(Spacer(1, 4 * mm))
                title_seen = True
            else:
                heading_counter += 1
                add_heading(story, styles, min(level - 1, 3), text_value, heading_counter)
            continue

        if line == "---":
            flush_paragraph()
            story.append(Spacer(1, 3 * mm))
            continue

        if not line.strip():
            flush_paragraph()
            continue

        bullet = re.match(r"^-\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(bullet.group(1)),
                    styles["ChineseBullet"],
                    bulletText="•",
                )
            )
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            flush_paragraph()
            story.append(
                Paragraph(
                    inline_markup(numbered.group(2)),
                    styles["ChineseBullet"],
                    bulletText=f"{numbered.group(1)}.",
                )
            )
            continue

        quote = re.match(r"^>\s?(.*)$", line)
        if quote:
            flush_paragraph()
            story.append(Paragraph(inline_markup(quote.group(1)), styles["ChineseQuote"]))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return story


def insert_toc(story, styles):
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOC1",
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            leftIndent=0,
            firstLineIndent=0,
            textColor=colors.HexColor("#163a63"),
        ),
        ParagraphStyle(
            name="TOC2",
            fontName="STSong-Light",
            fontSize=9.5,
            leading=14,
            leftIndent=12,
            firstLineIndent=0,
            textColor=colors.HexColor("#365f7d"),
        ),
        ParagraphStyle(
            name="TOC3",
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            leftIndent=24,
            firstLineIndent=0,
            textColor=colors.HexColor("#4a6478"),
        ),
    ]

    title = Paragraph("文档目录", styles["Heading1"])
    title._bookmark_name = "document-toc"
    insertion = [PageBreak(), title, Spacer(1, 3 * mm), toc, PageBreak()]
    story[2:2] = insertion


def main():
    styles = make_styles()
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), styles)
    insert_toc(story, styles)

    doc = TutorialDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="RCBF_SAC 5.2 初学者逐行讲解",
        author="Mod-RL-RCBF project tutorial",
        subject="Python, PyTorch and SAC code walkthrough for beginners",
    )
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
