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
    """
    Extract table data from PDF page, specifically targeting the Particulars column.
    """
    # Get text with detailed positioning
    text_dict = page.get_text("dict")
    
    # Collect all text blocks with positions
    text_blocks = []
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        text_blocks.append({
                            "text": span["text"].strip(),
                            "bbox": span["bbox"],
                            "x": span["bbox"][0],
                            "y": span["bbox"][1],
                            "font_size": span["size"]
                        })
    
    # Sort by Y position (top to bottom), then X position (left to right)
    text_blocks.sort(key=lambda x: (x["y"], x["x"]))
    
    # Find table header and structure
    header_row = []
    particulars_column_x = None
    
    # Look for table headers
    for block in text_blocks:
        text = block["text"].upper()
        if "PARTICULARS" in text:
            particulars_column_x = block["x"]
            break
        elif "S#" in text or "QTY" in text or "UNIT" in text:
            header_row.append(block)
    
    # If we found header, determine column boundaries
    if particulars_column_x is None and header_row:
        # Try to estimate particulars column position
        header_row.sort(key=lambda x: x["x"])
        if len(header_row) >= 2:
            particulars_column_x = header_row[1]["x"]  # Usually second column
    
    # Extract table rows
    table_data = []
    current_y = None
    current_row = []
    
    for block in text_blocks:
        # Skip header area
        if block["y"] < 200:  # Adjust based on your PDF structure
            continue
            
        # Group blocks by approximate Y position (same row)
        if current_y is None or abs(block["y"] - current_y) < 10:
            current_row.append(block)
            current_y = block["y"]
        else:
            # Process completed row
            if current_row:
                table_data.append(process_table_row(current_row, particulars_column_x))
            current_row = [block]
            current_y = block["y"]
    
    # Process last row
    if current_row:
        table_data.append(process_table_row(current_row, particulars_column_x))
    
    # Filter and clean data
    particulars_data = []
    for row in table_data:
        if row and row.get("particulars") and len(row["particulars"]) > 10:  # Filter meaningful entries
            particulars_data.append(row)
    
    return particulars_data

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

def create_high_quality_snippet(page, bbox, padding=20, zoom_factor=4.0, quality_enhance=True):
    """
    Create a high-quality snippet image from the specified bounding box.
    
    Args:
        page: PyMuPDF page object
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        padding: Padding around the bounding box
        zoom_factor: Zoom level for rendering (higher = better quality)
        quality_enhance: Whether to apply image enhancement filters
    """
    # Add padding to the bounding box
    page_rect = page.rect
    padded_rect = fitz.Rect(
        max(bbox[0] - padding, 0),
        max(bbox[1] - padding, 0),
        min(bbox[2] + padding, page_rect.width),
        min(bbox[3] + padding, page_rect.height)
    )
    
    # Create transformation matrix with high zoom for better quality
    matrix = fitz.Matrix(zoom_factor, zoom_factor)
    
    # Render the region to a pixmap with high quality settings
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
    if quality_enhance:
        pil_image = enhance_image_quality(pil_image)
    
    # Optional: Resize to reasonable dimensions while maintaining quality
    # This prevents extremely large images while keeping quality high
    max_width = 1200
    max_height = 800
    
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
        "file_size_kb": len(img_bytes) / 1024
    }

def create_snippet_image(page, bbox, padding=20):
    """
    Create a high-quality snippet image (wrapper for backward compatibility).
    """
    return create_high_quality_snippet(page, bbox, padding, zoom_factor=4.0, quality_enhance=True)

def create_ultra_high_quality_snippet(page, bbox, padding=20):
    """
    Create ultra high-quality snippet for premium use cases.
    """
    return create_high_quality_snippet(page, bbox, padding, zoom_factor=6.0, quality_enhance=True)

@app.post("/extract-particulars")
async def extract_particulars(file: UploadFile, high_quality: bool = True):
    """
    Extract particulars data from PDF with high-quality snippet images for Google Sheets.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        all_particulars = []
        
        for page_num, page in enumerate(doc):
            # Extract table data
            table_data = extract_table_data(page)
            
            for row in table_data:
                if row:
                    # Create high-quality snippet image for this row
                    if high_quality:
                        snippet = create_high_quality_snippet(page, row["full_row_bbox"])
                    else:
                        snippet = create_snippet_image(page, row["full_row_bbox"])
                    
                    particulars_entry = {
                        "page": page_num + 1,
                        "s_number": row["s_number"],
                        "particulars": row["particulars"],
                        "qty": row["qty"],
                        "snippet_image": snippet["image_url"],
                        "snippet_base64": snippet["image_data"],
                        "image_dimensions": snippet["dimensions"],
                        "file_size_kb": snippet.get("file_size_kb", 0),
                        "coordinates": {
                            "x0": row["bbox"][0] if row["bbox"] else 0,
                            "y0": row["bbox"][1] if row["bbox"] else 0,
                            "x1": row["bbox"][2] if row["bbox"] else 0,
                            "y1": row["bbox"][3] if row["bbox"] else 0
                        }
                    }
                    
                    all_particulars.append(particulars_entry)
        
        doc.close()
        
        return {
            "success": True,
            "total_items": len(all_particulars),
            "data": all_particulars,
            "google_sheets_ready": True,
            "quality_settings": {
                "high_quality": high_quality,
                "zoom_factor": 4.0 if high_quality else 2.0,
                "enhancements_applied": high_quality
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/extract-particulars-for-sheets")
async def extract_particulars_for_sheets(file: UploadFile, quality_level: str = "high"):
    """
    Extract particulars data formatted specifically for Google Sheets integration.
    
    Args:
        quality_level: "standard", "high", or "ultra" for different quality levels
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        sheets_data = []
        
        for page_num, page in enumerate(doc):
            table_data = extract_table_data(page)
            
            for row in table_data:
                if row:
                    # Create snippet image based on quality level
                    if quality_level == "ultra":
                        snippet = create_ultra_high_quality_snippet(page, row["full_row_bbox"])
                    elif quality_level == "high":
                        snippet = create_high_quality_snippet(page, row["full_row_bbox"])
                    else:
                        snippet = create_snippet_image(page, row["full_row_bbox"])
                    
                    # Format for Google Sheets
                    sheets_row = {
                        "S_Number": row["s_number"],
                        "Particulars": row["particulars"],
                        "Quantity": row["qty"],
                        "Page": page_num + 1,
                        "Image_Note": f"=IMAGE(\"{snippet['image_url']}\")",  # Google Sheets IMAGE formula
                        "Snippet_URL": snippet["image_url"],
                        "Hover_Text": f"Product: {row['particulars']} | Qty: {row['qty']} | Page: {page_num + 1}",
                        "Image_Quality": quality_level,
                        "Image_Size_KB": snippet.get("file_size_kb", 0),
                        "Image_Dimensions": f"{snippet['dimensions']['width']}x{snippet['dimensions']['height']}"
                    }
                    
                    sheets_data.append(sheets_row)
        
        doc.close()
        
        return {
            "success": True,
            "sheets_data": sheets_data,
            "total_rows": len(sheets_data),
            "columns": ["S_Number", "Particulars", "Quantity", "Page", "Image_Note", "Snippet_URL", "Hover_Text", "Image_Quality", "Image_Size_KB", "Image_Dimensions"],
            "quality_settings": {
                "level": quality_level,
                "total_size_mb": sum(row.get("Image_Size_KB", 0) for row in sheets_data) / 1024
            }
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
        if quality_level == "ultra":
            snippet = create_ultra_high_quality_snippet(selected_page, bbox, padding)
        elif quality_level == "high":
            snippet = create_high_quality_snippet(selected_page, bbox, padding)
        else:
            snippet = create_snippet_image(selected_page, bbox, padding)
        
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
                "description": "Basic quality, smaller file size",
                "recommended_for": "Quick previews, low bandwidth"
            },
            "high": {
                "zoom_factor": 4.0,
                "enhancements": True,
                "description": "High quality with image enhancements",
                "recommended_for": "Google Sheets, general use"
            },
            "ultra": {
                "zoom_factor": 6.0,
                "enhancements": True,
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


# Add these additional endpoints to your existing FastAPI code

@app.post("/get-pdf-pages")
async def get_pdf_pages(file: UploadFile):
    """
    Get all PDF pages as images with dimensions for coordinate reference.
    Useful for manual coordinate selection in n8n workflows.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        pages_info = []
        
        for page_num, page in enumerate(doc):
            # Render page as image for reference
            matrix = fitz.Matrix(2, 2)  # 2x zoom for clear reference
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            page_info = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "image_url": f"data:image/png;base64,{img_base64}",
                "image_dimensions": {
                    "width": pix.width,
                    "height": pix.height
                }
            }
            pages_info.append(page_info)
        
        doc.close()
        
        return {
            "success": True,
            "total_pages": len(pages_info),
            "pages": pages_info
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/find-text-coordinates")
async def find_text_coordinates(
    file: UploadFile,
    search_text: str = Form(...),
    page_number: int = Form(None),
    case_sensitive: bool = Form(False)
):
    """
    Find coordinates of specific text in PDF.
    Useful for locating specific items by text search.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        text_locations = []
        
        # Search in specific page or all pages
        pages_to_search = [page_number - 1] if page_number else range(len(doc))
        
        for page_idx in pages_to_search:
            if page_idx < 0 or page_idx >= len(doc):
                continue
                
            page = doc[page_idx]
            
            # Search for text
            if case_sensitive:
                text_instances = page.search_for(search_text)
            else:
                text_instances = page.search_for(search_text, flags=fitz.TEXT_IGNORE_CASE)
            
            for rect in text_instances:
                text_locations.append({
                    "page": page_idx + 1,
                    "text": search_text,
                    "coordinates": {
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1
                    },
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1]
                })
        
        doc.close()
        
        return {
            "success": True,
            "search_text": search_text,
            "total_found": len(text_locations),
            "locations": text_locations
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/get-table-structure")
async def get_table_structure(file: UploadFile, page_number: int = Form(1)):
    """
    Analyze table structure and return all detected cells with coordinates.
    Useful for understanding the table layout.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        if page_number < 1 or page_number > len(doc):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid page number. PDF has {len(doc)} pages."}
            )
        
        page = doc[page_number - 1]
        
        # Get all text blocks with positions
        text_dict = page.get_text("dict")
        all_text_blocks = []
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            all_text_blocks.append({
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
        all_text_blocks.sort(key=lambda x: (x["coordinates"]["y0"], x["coordinates"]["x0"]))
        
        doc.close()
        
        return {
            "success": True,
            "page": page_number,
            "total_text_blocks": len(all_text_blocks),
            "text_blocks": all_text_blocks,
            "usage_note": "Use these coordinates with /get-snippet-image endpoint"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/extract-coordinates-only")
async def extract_coordinates_only(file: UploadFile):
    """
    Extract only the coordinates of detected table rows without creating images.
    Lightweight endpoint for coordinate detection.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        all_coordinates = []
        
        for page_num, page in enumerate(doc):
            # Extract table data (reuse existing function)
            table_data = extract_table_data(page)
            
            for row in table_data:
                if row:
                    coordinate_entry = {
                        "page": page_num + 1,
                        "s_number": row["s_number"],
                        "particulars": row["particulars"],
                        "qty": row["qty"],
                        "text_coordinates": {
                            "x0": row["bbox"][0] if row["bbox"] else 0,
                            "y0": row["bbox"][1] if row["bbox"] else 0,
                            "x1": row["bbox"][2] if row["bbox"] else 0,
                            "y1": row["bbox"][3] if row["bbox"] else 0
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