from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import fitz  # PyMuPDF
import io
import re
import base64
from typing import List, Dict, Optional
import json
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

app = FastAPI()

def extract_table_data(page):

    words = page.get_text("words")  
    words.sort(key=lambda w: (w[1], w[0]))  
    
    rows = []
    current_row_y = None
    current_row = []

    for word in words:
        x0, y0, x1, y1, text = word[:5]
        if current_row_y is None or abs(y0 - current_row_y) <= 2:
            current_row.append(word)
            current_row_y = y0
        else:
            rows.append(current_row)
            current_row = [word]
            current_row_y = y0
    if current_row:
        rows.append(current_row)

    extracted_rows = []
    for row in rows:
        texts = [w[4] for w in row]
        bbox = [min(w[0] for w in row), min(w[1] for w in row),
                max(w[2] for w in row), max(w[3] for w in row)]
        
        extracted_rows.append({
            "text": " ".join(texts),
            "bbox": bbox,
            "words": row,
            "full_row_bbox": bbox  # Optional - same as bbox here, but can be adjusted
        })

    return extracted_rows


def process_table_row(row_blocks, particulars_column_x):
    """
    Process a single table row to extract relevant data.
    """
    row_blocks.sort(key=lambda x: x["x"])
    
    # Find particulars text (usually the longest text in the row)
    particulars_text = ""
    particulars_bbox = None
    s_number = ""
    qty = ""
    
    # Extract S# (serial number) - usually first column
    if row_blocks and re.match(r'^\d+$', row_blocks[0]["text"]):
        s_number = row_blocks[0]["text"]
    
    # Find particulars - look for product description patterns
    for block in row_blocks:
        text = block["text"]
        # Identify product descriptions (contains oil, engine, etc.)
        if (len(text) > 15 and 
            any(keyword in text.upper() for keyword in ["OIL", "ENGINE", "GEAR", "HYDRAULIC", "MOTOR"])):
            if len(text) > len(particulars_text):  # Take the longest matching text
                particulars_text = text
                particulars_bbox = block["bbox"]
    
    # Extract quantity (look for patterns like "399.00", "192.00")
    for block in row_blocks:
        text = block["text"]
        if re.match(r'^\d+\.?\d*$', text) and len(text) >= 3:
            qty = text
            break
    
    if particulars_text:
        return {
            "s_number": s_number,
            "particulars": particulars_text,
            "qty": qty,
            "bbox": particulars_bbox,
            "full_row_bbox": calculate_row_bbox(row_blocks)
        }
    
    return None

def calculate_row_bbox(row_blocks):
    """Calculate bounding box for entire row."""
    if not row_blocks:
        return None
    
    min_x = min(block["bbox"][0] for block in row_blocks)
    min_y = min(block["bbox"][1] for block in row_blocks)
    max_x = max(block["bbox"][2] for block in row_blocks)
    max_y = max(block["bbox"][3] for block in row_blocks)
    
    return [min_x, min_y, max_x, max_y]

def enhance_image_quality(pil_image):
    """
    Apply multiple enhancement techniques to improve image quality.
    """
    # Convert to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 1. Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)  # Increase sharpness by 50%
    
    # 2. Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase contrast by 20%
    
    # 3. Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.1)  # Increase brightness by 10%
    
    # 4. Apply unsharp mask filter for better text clarity
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return pil_image

def create_quality_snippet(page, bbox, padding=20, quality_level="high"):
    """
    Create snippet image with different quality levels.
    
    Args:
        quality_level: "standard", "high", or "ultra"
    """
    # Quality settings
    quality_settings = {
        "standard": {"zoom": 2.0, "enhance": False, "max_width": 800, "max_height": 600},
        "high": {"zoom": 4.0, "enhance": True, "max_width": 1200, "max_height": 800},
        "ultra": {"zoom": 6.0, "enhance": True, "max_width": 1600, "max_height": 1200}
    }
    
    settings = quality_settings.get(quality_level, quality_settings["high"])
    
    # Add padding to the bounding box
    page_rect = page.rect
    padded_rect = fitz.Rect(
        max(bbox[0] - padding, 0),
        max(bbox[1] - padding, 0),
        min(bbox[2] + padding, page_rect.width),
        min(bbox[3] + padding, page_rect.height)
    )
    
    matrix = fitz.Matrix(settings["zoom"], settings["zoom"])
    
    pix = page.get_pixmap(
        clip=padded_rect, 
        matrix=matrix,
        alpha=False,  # No transparency for better file size
        colorspace=fitz.csRGB  # Ensure RGB colorspace
    )
    
    # Convert to PIL Image for enhancement
    img_bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    # Apply quality enhancements
    if settings["enhance"]:
        pil_image = enhance_image_quality(pil_image)
    
    # Resize to reasonable dimensions while maintaining quality
    max_width = settings["max_width"]
    max_height = settings["max_height"]
    
    if pil_image.width > max_width or pil_image.height > max_height:
        # Calculate resize ratio maintaining aspect ratio
        width_ratio = max_width / pil_image.width
        height_ratio = max_height / pil_image.height
        resize_ratio = min(width_ratio, height_ratio)
        
        new_width = int(pil_image.width * resize_ratio)
        new_height = int(pil_image.height * resize_ratio)
        
        # Use LANCZOS resampling for high quality resize
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert back to bytes with high quality settings
    output_buffer = io.BytesIO()
    pil_image.save(
        output_buffer, 
        format='PNG', 
        optimize=True,
        compress_level=6  # Good compression without quality loss
    )
    
    img_bytes = output_buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return {
        "image_data": img_base64,
        "image_url": f"data:image/png;base64,{img_base64}",
        "dimensions": {"width": pil_image.width, "height": pil_image.height},
        "file_size_kb": len(img_bytes) / 1024,
        "quality_level": quality_level
    }

def search_multiline_text(page, query, max_line_gap=50):
    """
    Search for text that might span multiple lines in a PDF page with exact word order matching.
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

# CORE ENDPOINTS (PRESERVED)

@app.post("/extract-coordinates-only")
async def extract_coordinates_only(file: UploadFile):
    """
    Dynamic invoice processor to extract row-wise coordinates from any invoice PDF.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        all_coordinates = []

        for page_num, page in enumerate(doc):
            table_data = extract_table_data(page)
            
            for row in table_data:
                coordinate_entry = {
                    "page": page_num + 1,
                    "text": row["text"],
                    "text_coordinates": {
                        "x0": row["bbox"][0],
                        "y0": row["bbox"][1],
                        "x1": row["bbox"][2],
                        "y1": row["bbox"][3]
                    },
                    "row_coordinates": {
                        "x0": row["full_row_bbox"][0],
                        "y0": row["full_row_bbox"][1],
                        "x1": row["full_row_bbox"][2],
                        "y1": row["full_row_bbox"][3]
                    }
                }
                all_coordinates.append(coordinate_entry)
        
        doc.close()

        return {
            "success": True,
            "total_items": len(all_coordinates),
            "coordinates": all_coordinates,
            "note": "Use these coordinates with /get-snippet-image to generate images"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

    
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
    """
    Get high-quality snippet image for specific coordinates.
    
    Args:
        quality_level: "standard", "high", or "ultra"
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        if page < 1 or page > len(doc):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid page number"}
            )
        
        selected_page = doc[page - 1]
        bbox = [x0, y0, x1, y1]
        
        # Create snippet based on quality level
        snippet = create_quality_snippet(selected_page, bbox, padding, quality_level)
        
        # Return as streaming image
        img_bytes = base64.b64decode(snippet["image_data"])
        return StreamingResponse(
            io.BytesIO(img_bytes), 
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=snippet_p{page}_quality_{quality_level}.png",
                "X-Image-Quality": quality_level,
                "X-Image-Size": f"{snippet['dimensions']['width']}x{snippet['dimensions']['height']}",
                "X-File-Size-KB": str(snippet.get("file_size_kb", 0))
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# UTILITY ENDPOINTS

@app.post("/search-text")
async def search_text(
    file: UploadFile,
    query: str = Form(...),
    search_type: str = Form("expanded"),  # "normal", "expanded"
    padding: float = Form(30.0)
):
    """
    Universal text search endpoint combining normal and expanded search functionality.
    
    Args:
        search_type: "normal" returns all matches, "expanded" returns best match with padding
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        if search_type == "normal":
            # Return all matches
            matches = []
            for i, page in enumerate(doc):
                # Regular search
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
                
                # Multi-line search
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
            
            return {"success": True, "matches": matches}
        
        else:  # expanded search
            # Return best match with padding
            best_match = None
            best_match_type = None
            best_match_rect = None
            best_match_score = 0
            best_match_page = None

            query_words = [word.upper() for word in query.split() if word.strip()]
            if not query_words:
                return {"error": "Empty or invalid query"}

            for i, page in enumerate(doc):
                width, height = page.rect.width, page.rect.height

                # Single-line match check
                rects = page.search_for(query)
                for r in rects:
                    match_score = len(query)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_type = "single_line"
                        best_match_rect = [
                            max(r.x0 - padding, 0),
                            max(r.y0 - padding, 0),
                            min(r.x1 + padding, width),
                            min(r.y1 + padding, height)
                        ]
                        best_match_page = i + 1

                # Multi-line match check
                multiline_rects = search_multiline_text(page, query)
                for r in multiline_rects:
                    rect_text = page.get_textbox(fitz.Rect(*r))
                    match_words = [word.upper() for word in rect_text.split() if word.strip()]
                    for start_idx in range(len(match_words) - len(query_words) + 1):
                        if match_words[start_idx:start_idx + len(query_words)] == query_words:
                            match_score = len(query_words)
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_match_type = "multi_line"
                                best_match_rect = [
                                    max(r[0] - padding, 0),
                                    max(r[1] - padding, 0),
                                    min(r[2] + padding, width),
                                    min(r[3] + padding, height)
                                ]
                                best_match_page = i + 1
                            break

            if best_match_rect:
                return {
                    "success": True,
                    "page": best_match_page,
                    "x0": best_match_rect[0],
                    "y0": best_match_rect[1],
                    "x1": best_match_rect[2],
                    "y1": best_match_rect[3],
                    "type": best_match_type
                }
            else:
                return {"success": False, "message": "No match found"}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/get-pdf-info")
async def get_pdf_info(
    file: UploadFile,
    include_pages: bool = Form(False),
    page_number: int = Form(None)
):
    """
    Get PDF information including pages, dimensions, and text structure.
    
    Args:
        include_pages: Whether to include page images
        page_number: Specific page to analyze (if not provided, analyzes all pages)
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        pdf_info = {
            "success": True,
            "total_pages": len(doc),
            "pages": []
        }
        
        # Determine which pages to process
        if page_number:
            if page_number < 1 or page_number > len(doc):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid page number. PDF has {len(doc)} pages."}
                )
            pages_to_process = [page_number - 1]
        else:
            pages_to_process = range(len(doc))
        
        for page_idx in pages_to_process:
            page = doc[page_idx]
            page_info = {
                "page_number": page_idx + 1,
                "width": page.rect.width,
                "height": page.rect.height
            }
            
            # Include page image if requested
            if include_pages:
                matrix = fitz.Matrix(2, 2)  # 2x zoom for clear reference
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                page_info.update({
                    "image_url": f"data:image/png;base64,{img_base64}",
                    "image_dimensions": {
                        "width": pix.width,
                        "height": pix.height
                    }
                })
            
            # Add text structure analysis
            text_dict = page.get_text("dict")
            text_blocks = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                text_blocks.append({
                                    "text": span["text"].strip(),
                                    "coordinates": {
                                        "x0": span["bbox"][0],
                                        "y0": span["bbox"][1],
                                        "x1": span["bbox"][2],
                                        "y1": span["bbox"][3]
                                    },
                                    "font_size": span["size"],
                                    "font_name": span["font"]
                                })
            
            # Sort by position
            text_blocks.sort(key=lambda x: (x["coordinates"]["y0"], x["coordinates"]["x0"]))
            
            page_info["text_blocks"] = text_blocks
            page_info["total_text_blocks"] = len(text_blocks)
            
            pdf_info["pages"].append(page_info)
        
        doc.close()
        return pdf_info
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/snip-region")
async def snip_region(
    file: UploadFile,
    page: int = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    x1: float = Form(...),
    y1: float = Form(...),
    padding: float = Form(0.0)
):
    """
    Simple region snipping endpoint for quick image extraction.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")

        # Validate page number
        if page < 1 or page > len(doc):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid page number"}
            )

        selected_page = doc[page - 1]
        width, height = selected_page.rect.width, selected_page.rect.height

        # Apply padding
        padded_x0 = max(x0 - padding, 0)
        padded_y0 = max(y0 - padding, 0)
        padded_x1 = min(x1 + padding, width)
        padded_y1 = min(y1 + padding, height)

        rect = fitz.Rect(padded_x0, padded_y0, padded_x1, padded_y1)

        # Render clipped region to pixmap (image)
        pix = selected_page.get_pixmap(clip=rect, dpi=150)
        img_bytes = pix.tobytes("png")

        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/quality-settings")
async def get_quality_settings():
    """
    Get available quality settings and their descriptions.
    """
    return {
        "quality_levels": {
            "standard": {
                "zoom_factor": 2.0,
                "enhancements": False,
                "max_dimensions": "800x600",
                "description": "Basic quality, smaller file size",
                "recommended_for": "Quick previews, low bandwidth"
            },
            "high": {
                "zoom_factor": 4.0,
                "enhancements": True,
                "max_dimensions": "1200x800",
                "description": "High quality with image enhancements",
                "recommended_for": "Google Sheets, general use"
            },
            "ultra": {
                "zoom_factor": 6.0,
                "enhancements": True,
                "max_dimensions": "1600x1200",
                "description": "Ultra high quality for premium use cases",
                "recommended_for": "Print quality, detailed analysis"
            }
        },
        "enhancements_applied": [
            "Sharpness enhancement (+50%)",
            "Contrast enhancement (+20%)",
            "Brightness adjustment (+10%)",
            "Unsharp mask filter for text clarity",
            "LANCZOS resampling for resizing"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)