"""Build the fictional policy PDF. Requires reportlab as an authoring tool only."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sample/encryption_key_management_policy.json"
OUTPUT = ROOT / "data/sample/encryption_key_management_policy.pdf"

def build(source_path=SOURCE, output_path=OUTPUT):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PolicyBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="PolicySection", fontName="Helvetica-Bold", fontSize=11, leading=14, spaceAfter=5, textColor=colors.HexColor("#17465c")))
    story = []
    for index, page in enumerate(source["pages"]):
        if index:
            story.append(PageBreak())
        story.append(Paragraph("CloudDesk Technologies", styles["PolicySection"]))
        story.append(Paragraph(source["title"], styles["Title"]))
        story.append(Paragraph(escape(page["heading"]), styles["Heading2"]))
        story.append(Spacer(1, 8))
        for section in page["sections"]:
            story.append(Paragraph(escape(section["heading"]), styles["PolicySection"]))
            for paragraph in section["text"].split("\n\n"):
                story.append(Paragraph(escape(paragraph), styles["PolicyBody"]))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#425463"))
        canvas.drawString(48, 30, "SYNTHETIC / DEMO | CD-ENC-001 v1.0 | Requires user review")
        canvas.drawRightString(A4[0]-48, 30, f"Page {doc.page} of {len(source['pages'])}")
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=48, leftMargin=48,
        topMargin=40, bottomMargin=50, invariant=1, title=source["title"] + " - SYNTHETIC DEMO",
        author="CloudDesk Technologies (fictional)", subject="Model-authored proposed encryption policy; review_status: requires_user_review; approval_status: not_recorded")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return Path(output_path)

if __name__ == "__main__":
    build()
