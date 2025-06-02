from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
import io
import re

app = FastAPI()

def search_multiline_text(page, query, max_line_gap=50):
    """
    Search for text that might span multiple lines in a PDF page with exact word order matching.
    
    Args:
        page: fitz.Page object
        query: search string (must match exactly in same word order)
        max_line_gap: maximum vertical distance between lines to consider them connected
    
    Returns:
        List of rectangles that encompass the matched text across lines
    """
    # Get all text blocks from the page with their positions
    text_dict = page.get_text("dict")
    
    # Extract text with coordinates, preserving original text structure
    text_segments = []
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                
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
    
    # Normalize query: clean whitespace but preserve word order
    query_words = [word.upper() for word in query.split() if word.strip()]
    if not query_words:
        return []
    
    matches = []
    
    # Search for the query in combinations of consecutive lines
    for i in range(len(text_segments)):
        combined_text = ""
        combined_rects = []
        
        # Try combining up to 6 consecutive lines
        for j in range(i, min(i + 6, len(text_segments))):
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
            
            # Normalize combined text: split into words and clean
            combined_words = [word.upper() for word in combined_text.split() if word.strip()]
            
            # Check for exact word sequence match
            if len(combined_words) >= len(query_words):
                # Look for the exact sequence of query words in the combined text
                for start_idx in range(len(combined_words) - len(query_words) + 1):
                    match_found = True
                    for k, query_word in enumerate(query_words):
                        if combined_words[start_idx + k] != query_word:
                            match_found = False
                            break
                    
                    if match_found:
                        # Calculate bounding rectangle that encompasses all matched lines
                        if combined_rects:
                            min_x0 = min(rect[0] for rect in combined_rects)
                            min_y0 = min(rect[1] for rect in combined_rects)
                            max_x1 = max(rect[2] for rect in combined_rects)
                            max_y1 = max(rect[3] for rect in combined_rects)
                            
                            # Check if this match overlaps with existing matches
                            is_duplicate = False
                            for existing_match in matches:
                                if (abs(existing_match[0] - min_x0) < 5 and 
                                    abs(existing_match[1] - min_y0) < 5):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                matches.append([min_x0, min_y0, max_x1, max_y1])
                        break  # Found exact match, no need to continue checking this combination
    
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

@app.post("/search-multiline-only")
async def search_multiline_only(file: UploadFile, query: str = Form(...)):
    """
    Search for exact text matches with intelligent single/multi-line detection.
    Returns single-line matches if found, otherwise returns multi-line matches.
    Prioritizes exact matches regardless of line distribution.
    """
    pdf_bytes = await file.read()
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream, filetype="pdf")

    matches = []
    query_words = [word.upper() for word in query.split() if word.strip()]
    
    for i, page in enumerate(doc):
        # First: Check for exact single-line matches
        single_line_matches = []
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_bbox = None
                    
                    for span in line["spans"]:
                        span_text = span["text"]
                        line_text += span_text
                        
                        # Calculate bounding box for the entire line
                        if line_bbox is None:
                            line_bbox = list(span["bbox"])
                        else:
                            line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                            line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                            line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                            line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                    
                    if line_text.strip() and line_bbox:
                        line_words = [word.upper() for word in line_text.split() if word.strip()]
                        
                        # Check for exact word sequence in this single line
                        if len(line_words) >= len(query_words):
                            for start_idx in range(len(line_words) - len(query_words) + 1):
                                if line_words[start_idx:start_idx + len(query_words)] == query_words:
                                    single_line_matches.append({
                                        "page": i + 1,
                                        "x0": line_bbox[0],
                                        "y0": line_bbox[1],
                                        "x1": line_bbox[2],
                                        "y1": line_bbox[3],
                                        "type": "single_line",
                                        "query": query,
                                        "matched_text": line_text.strip()
                                    })
                                    break
        
        # If single-line matches found, use them
        if single_line_matches:
            matches.extend(single_line_matches)
        else:
            # If no single-line matches, search for multi-line matches
            multiline_rects = search_multiline_text(page, query)
            for r in multiline_rects:
                matches.append({
                    "page": i + 1,
                    "x0": r[0],
                    "y0": r[1],
                    "x1": r[2],
                    "y1": r[3],
                    "type": "multi_line",
                    "query": query
                })

    return {"matches": matches, "query": query, "total_matches": len(matches)}