from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from xml.sax.saxutils import escape

text = open("report.md", encoding="utf-8").read()

styles = getSampleStyleSheet()

title = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=8,
)

heading = ParagraphStyle(
    "ReportHeading",
    parent=styles["Heading1"],
    fontSize=13,
    leading=16,
    spaceBefore=8,
    spaceAfter=5,
)

body = ParagraphStyle(
    "ReportBody",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10,
    spaceAfter=4,
)

doc = SimpleDocTemplate(
    "report.pdf",
    pagesize=A4,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36,
)

story = []

for line in text.splitlines():
    line = line.strip()

    if line.startswith("# "):
        story.append(Paragraph(escape(line[2:]), title))

    elif line.startswith("## "):
        story.append(Paragraph(escape(line[3:]), heading))

    elif line.startswith("### "):
        story.append(Paragraph(escape(line[4:]), styles["Heading3"]))

    elif line.startswith("|") and "---" not in line:
        cells = [
            escape(cell.strip())
            for cell in line.strip("|").split("|")
        ]
        table = Table([[Paragraph(cell, body) for cell in cells]])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)

    elif line:
        clean = line.replace("**", "")
        story.append(Paragraph(escape(clean), body))

    else:
        story.append(Spacer(1, 3))

doc.build(story)

print("Created report.pdf")
