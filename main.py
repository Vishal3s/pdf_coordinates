from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
import re

app = FastAPI()

def search_multiline_text(page, query, max_line_gap=50):
    """
    Search for text that might span multiple lines in a PDF page.
    
    Args:
        page: fitz.Page object
        query: search string
        max_line_gap: maximum vertical distance between lines to consider them connected
    
    Returns:
        List of rectangles that encompass the matched text across lines
    """
    # Get all text blocks from the page with their positions
    text_dict = page.get_text("dict")
    
    # Extract text with coordinates
    text_segments = []
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                char_bboxes = []
                
                for span in line["spans"]:
                    span_text = span["text"]
                    line_text += span_text
                    
                    # Calculate bounding box for the entire line
                    if line_bbox is None:
                        line_bbox = list(span["bbox"])
                    else:
                        line_bbox[0] = min(line_bbox[0], span["bbox"][0])  # x0
                        line_bbox[1] = min(line_bbox[1], span["bbox"][1])  # y0
                        line_bbox[2] = max(line_bbox[2], span["bbox"][2])  # x1
                        line_bbox[3] = max(line_bbox[3], span["bbox"][3])  # y1
                
                if line_text.strip():
                    text_segments.append({
                        "text": line_text.strip(),
                        "bbox": line_bbox,
                        "y_center": (line_bbox[1] + line_bbox[3]) / 2
                    })
    
    # Sort segments by vertical position (top to bottom)
    text_segments.sort(key=lambda x: x["y_center"])
    
    # Clean the query for better matching
    query_cleaned = re.sub(r'\s+', ' ', query.strip().upper())
    matches = []
    
    # Search for the query in combinations of consecutive lines
    for i in range(len(text_segments)):
        combined_text = ""
        combined_rects = []
        
        # Try combining up to 5 consecutive lines
        for j in range(i, min(i + 5, len(text_segments))):
            current_segment = text_segments[j]
            
            # Check if this line is close enough to the previous one
            if j > i:
                prev_segment = text_segments[j-1]
                vertical_gap = current_segment["y_center"] - prev_segment["y_center"]
                if vertical_gap > max_line_gap:
                    break
            
            # Add current line text
            if combined_text:
                combined_text += " "
            combined_text += current_segment["text"]
            combined_rects.append(current_segment["bbox"])
            
            # Clean combined text for comparison
            combined_text_cleaned = re.sub(r'\s+', ' ', combined_text.upper())
            
            # Check if query matches the combined text
            if query_cleaned in combined_text_cleaned:
                # Calculate bounding rectangle that encompasses all matched lines
                if combined_rects:
                    min_x0 = min(rect[0] for rect in combined_rects)
                    min_y0 = min(rect[1] for rect in combined_rects)
                    max_x1 = max(rect[2] for rect in combined_rects)
                    max_y1 = max(rect[3] for rect in combined_rects)
                    
                    # Check if this match overlaps with existing matches
                    is_duplicate = False
                    for existing_match in matches:
                        if (abs(existing_match[0] - min_x0) < 10 and 
                            abs(existing_match[1] - min_y0) < 10):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        matches.append([min_x0, min_y0, max_x1, max_y1])
    
    return matches

@app.post("/search-normal")
async def search_normal(file: UploadFile, query: str = Form(...)):
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    matches = []
    for i, page in enumerate(doc):
        # First try regular search
        rects = page.search_for(query)
        for r in rects:
            matches.append({
                "page": i + 1,
                "x0": r.x0,
                "y0": r.y0,
                "x1": r.x1,
                "y1": r.y1,
                "type": "single_line"
            })
        
        # Then try multi-line search
        multiline_rects = search_multiline_text(page, query)
        for r in multiline_rects:
            matches.append({
                "page": i + 1,
                "x0": r[0],
                "y0": r[1],
                "x1": r[2],
                "y1": r[3],
                "type": "multi_line"
            })

    return {"matches": matches}

@app.post("/search-expanded")
async def search_expanded(file: UploadFile, query: str = Form(...)):
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    matches = []
    padding = 30  # You can tune this value

    for i, page in enumerate(doc):
        width, height = page.rect.width, page.rect.height
        
        # Regular search
        rects = page.search_for(query)
        for r in rects:
            matches.append({
                "page": i + 1,
                "x0": max(r.x0 - padding, 0),
                "y0": max(r.y0 - padding, 0),
                "x1": min(r.x1 + padding, width),
                "y1": min(r.y1 + padding, height),
                "type": "single_line"
            })
        
        # Multi-line search
        multiline_rects = search_multiline_text(page, query)
        for r in multiline_rects:
            matches.append({
                "page": i + 1,
                "x0": max(r[0] - padding, 0),
                "y0": max(r[1] - padding, 0),
                "x1": min(r[2] + padding, width),
                "y1": min(r[3] + padding, height),
                "type": "multi_line"
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

    # Ensure page number is within bounds
    if page < 1 or page > len(doc):
        return {"error": "Invalid page number"}

    selected_page = doc[page - 1]
    rect = fitz.Rect(x0, y0, x1, y1)

    # Render the selected region to a pixmap (image)
    pix = selected_page.get_pixmap(clip=rect)

    img_bytes = pix.tobytes("png")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")