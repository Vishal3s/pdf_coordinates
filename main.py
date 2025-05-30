from fastapi import FastAPI, UploadFile, Form
import fitz  # PyMuPDF
import io

app = FastAPI()

@app.post("/search")
async def search_in_pdf(file: UploadFile, query: str = Form(...)):
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    matches = []
    for i, page in enumerate(doc):
        rects = page.search_for(query)
        for r in rects:
            matches.append({
                "page": i + 1,
                "x0": r.x0,
                "y0": r.y0,
                "x1": r.x1,
                "y1": r.y1
            })
    return {"matches": matches}
