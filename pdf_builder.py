from typing import List, Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from outputs import _call_llm

def build_pdf_response(query: str, retrieved: Dict[str, Any], profile: Dict[str, Any], out_pdf_path: str = "report.pdf") -> Dict[str, Any]:
    """
    Generates a professional, structured PDF report with a summary and details.
    """
    docs = retrieved.get("documents", [])
    metas = retrieved.get("metadatas", [])
    
    #  Generate an Executive Summary with an LLM
    summary_context = "\n".join([f"- {doc}" for doc in docs])
    summary_prompt = f"""
    You are a business analyst. Your task is to write a concise "Executive Summary" section for a report.
    Analyze the provided context of conversation snippets and the user's original query.
    Your summary should directly answer the user's query based ONLY on the information in the snippets.

    User Query: "{query}"
    """
    executive_summary = _call_llm(summary_prompt, context=summary_context)

    #Building the PDF Document using ReportLab
    doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("AI-Generated Conversation Report", styles['h1']))
    story.append(Spacer(1, 12))

    # Conversation Metadata from Profile
    story.append(Paragraph(f"<b>Source File:</b> {profile.get('source_file', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Conversation Type:</b> {profile.get('conversation_type', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Overall Summary:</b> {profile.get('overall_summary', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 24))

 
    story.append(Paragraph("Executive Summary", styles['h2']))
    story.append(Paragraph(f"<i>Based on the query: '{query}'</i>", styles['Italic']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(executive_summary.replace("\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 24))


    story.append(Paragraph("Detailed Evidence", styles['h2']))
    story.append(Paragraph("<i>The following are the most relevant segments found in the transcript:</i>", styles['Italic']))
    story.append(Spacer(1, 12))

    for doc_text, meta in zip(docs, metas):
        speaker = meta.get("speaker", "Unknown")
        start_time = meta.get("start", 0)
        end_time = meta.get("end", 0)
        # Format the metadata and text for the report
        meta_info = f"<b>{speaker}</b> (from {start_time:.1f}s to {end_time:.1f}s):"
        story.append(Paragraph(meta_info, styles['Normal']))
        story.append(Paragraph(doc_text, styles['BodyText']))
        story.append(Spacer(1, 12))

    doc.build(story)
    
    return {"type": "pdf", "pdf_path": out_pdf_path}