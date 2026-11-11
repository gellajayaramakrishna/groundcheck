"""FastAPI backend: upload a PDF + question (+ optional answer). If no answer is given and a
GROQ_API_KEY is set, an LLM writes one. Every answer sentence is checked against the PDF."""
import io
import os

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pypdf import PdfReader

from src.checker import GroundChecker

app = FastAPI(title="GroundCheck")
checker = GroundChecker()  # loads models/groundcheck if trained, else the zero-shot NLI model
STATIC = os.path.join(os.path.dirname(__file__), "static")


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def ask_llm(document: str, question: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(400, "No answer given and GROQ_API_KEY is not set. Paste an answer instead.")
    prompt = (f"Answer the question using only this document.\n\nDocument:\n{document[:12000]}"
              f"\n\nQuestion: {question}")
    r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                   headers={"Authorization": f"Bearer {key}"},
                   json={"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                         "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
                   timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, f"LLM call failed: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": checker.model_path, "threshold": checker.threshold}


@app.post("/api/check")
async def check(pdf: UploadFile = File(...), question: str = Form(""), answer: str = Form("")):
    document = pdf_to_text(await pdf.read())
    if not document.strip():
        raise HTTPException(400, "Could not read any text from this PDF (is it a scanned image?).")
    if not answer.strip():
        answer = ask_llm(document, question)
    results = checker.check(document, answer, question=question or None)
    flagged = sum(not r["supported"] for r in results)
    return {"answer": answer, "sentences": results, "flagged": flagged, "total": len(results)}
