from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from PIL import Image
import pytesseract
import io
import base64
import fitz
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()


@app.post("/get-snippet-image")
async def get_snippet_image(
    file: UploadFile,
    page: int = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    x1: float = Form(...),
    y1: float = Form(...),
    padding: int = Form(20),
    quality_level: str = Form("high")
):
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")

        if page < 1 or page > len(doc):
            return JSONResponse(status_code=400, content={"error": "Invalid page number"})

        selected_page = doc[page - 1]
        page_rect = selected_page.rect
        bbox = fitz.Rect(
            max(x0 - padding, 0),
            max(y0 - padding, 0),
            min(x1 + padding, page_rect.width),
            min(y1 + padding, page_rect.height)
        )

        matrix = fitz.Matrix(4, 4)
        pix = selected_page.get_pixmap(clip=bbox, matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")

        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=snippet.png"}
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/get-row-by-query")
async def get_row_by_query(
    file: UploadFile,
    query: str = Form(...),
    page: Optional[int] = Form(None),
    padding: int = Form(5)
):
    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        results = []

        pages_to_search = [page - 1] if page else range(len(doc))

        for page_num in pages_to_search:
            current_page = doc[page_num]
            rects = current_page.search_for(query)

            for rect in rects:
                words = current_page.get_text("words")
                y_center = (rect.y0 + rect.y1) / 2
                y_thresh = max(10, (rect.y1 - rect.y0) * 1.5)
                row_words = [w[:4] for w in words if abs(((w[1] + w[3]) / 2) - y_center) <= y_thresh]

                if row_words:
                    x0 = min(w[0] for w in row_words)
                    y0 = min(w[1] for w in row_words)
                    x1 = max(w[2] for w in row_words)
                    y1 = max(w[3] for w in row_words)

                    padded = current_page.rect
                    padded_bbox = [
                        max(x0 - padding, 0),
                        max(y0 - padding, 0),
                        min(x1 + padding, padded.width),
                        min(y1 + padding, padded.height)
                    ]

                    row_text = current_page.get_textbox(fitz.Rect(*padded_bbox))

                    results.append({
                        "page": page_num + 1,
                        "query": query,
                        "found_at": {"x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1},
                        "full_row": {"x0": padded_bbox[0], "y0": padded_bbox[1],
                                     "x1": padded_bbox[2], "y1": padded_bbox[3], "text": row_text.strip()}
                    })

        doc.close()

        if not results:
            return JSONResponse(status_code=404, content={"success": False, "message": f"No matches for '{query}'"})
        return {"success": True, "matches": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/detect-coordinates-in-image")
async def detect_coordinates_in_image(file: UploadFile, query: str = Form(...), padding: int = Form(10)):
    import pytesseract
    from PIL import Image
    import io

    # Set Tesseract path if needed
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n = len(data['text'])
        query_words = query.strip().upper().split()
        matches = []

        for i in range(n - len(query_words) + 1):
            block_words = [data['text'][j].strip().upper() for j in range(i, i + len(query_words))]
            if block_words == query_words:
                # Coordinates of query match
                q_x0 = min(data['left'][i:i + len(query_words)])
                q_y0 = min(data['top'][i:i + len(query_words)])
                q_x1 = max(data['left'][j] + data['width'][j] for j in range(i, i + len(query_words)))
                q_y1 = max(data['top'][j] + data['height'][j] for j in range(i, i + len(query_words)))

                query_y_center = (q_y0 + q_y1) / 2

                # Find full row by checking all words within same y-level
                row_indices = []
                for k in range(n):
                    if data['text'][k].strip():
                        word_y_center = data['top'][k] + data['height'][k] / 2
                        if abs(word_y_center - query_y_center) <= 10:  # row alignment tolerance
                            row_indices.append(k)

                if row_indices:
                    row_x0 = min(data['left'][k] for k in row_indices)
                    row_y0 = min(data['top'][k] for k in row_indices)
                    row_x1 = max(data['left'][k] + data['width'][k] for k in row_indices)
                    row_y1 = max(data['top'][k] + data['height'][k] for k in row_indices)

                    matches.append({
                        "query_match": " ".join(block_words),
                        "query_coordinates": {"x0": q_x0, "y0": q_y0, "x1": q_x1, "y1": q_y1},
                        "full_row_coordinates": {
                            "x0": max(row_x0 - padding, 0),
                            "y0": max(row_y0 - padding, 0),
                            "x1": row_x1 + padding,
                            "y1": row_y1 + padding
                        },
                        "row_text": " ".join(data['text'][k] for k in row_indices)
                    })

        return {
            "success": True,
            "total_matches": len(matches),
            "matches": matches
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/get-image-snippet")
async def get_image_snippet(
    file: UploadFile,
    x0: int = Form(...),
    y0: int = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    padding: int = Form(10)
):
    """
    Crop a region from a JPG/PNG image based on coordinates.
    Returns the cropped region as a PNG image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = image.size

        # Apply padding safely
        x0_p = max(x0 - padding, 0)
        y0_p = max(y0 - padding, 0)
        x1_p = min(x1 + padding, width)
        y1_p = min(y1 + padding, height)

        # Crop the image
        cropped = image.crop((x0_p, y0_p, x1_p, y1_p))

        # Convert to bytes
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png", headers={
            "Content-Disposition": f"inline; filename=snippet.png"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
