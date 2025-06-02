from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import fitz  # PyMuPDF
import io

app = FastAPI()


@app.post("/search-normal")
async def search_normal(file: UploadFile, query: str = Form(...)):
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


@app.post("/search-expanded")
async def search_expanded(file: UploadFile, query: str = Form(...)):
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    matches = []
    padding = 30
    for i, page in enumerate(doc):
        rects = page.search_for(query)
        width, height = page.rect.width, page.rect.height

        for r in rects:
            matches.append({
                "page": i + 1,
                "x0": max(r.x0 - padding, 0),
                "y0": max(r.y0 - padding, 0),
                "x1": min(r.x1 + padding, width),
                "y1": min(r.y1 + padding, height)
            })

    return {"matches": matches}


@app.post("/snip-region")
async def snip_region(
    file: UploadFile,
    page: int = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    x1: float = Form(...),
    y1: float = Form(...)
):
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    if not (1 <= page <= len(doc)):
        return JSONResponse(content={"error": "Invalid page number"}, status_code=400)

    rect = fitz.Rect(x0, y0, x1, y1)
    pix = doc[page - 1].get_pixmap(clip=rect)

    img_bytes = pix.tobytes("png")
    img_stream = io.BytesIO(img_bytes)

    # Serve the image with a content-disposition to trigger download
    return StreamingResponse(
        img_stream,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename=snip_page{page}.png"
        }
    )
