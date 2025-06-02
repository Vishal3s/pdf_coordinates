from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
import fitz  # PyMuPDF
import io
import re
import base64
from typing import List, Dict, Optional
import json

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

def create_snippet_image(page, bbox, padding=20):
    """
    Create a snippet image from the specified bounding box.
    """
    # Add padding to the bounding box
    page_rect = page.rect
    padded_rect = fitz.Rect(
        max(bbox[0] - padding, 0),
        max(bbox[1] - padding, 0),
        min(bbox[2] + padding, page_rect.width),
        min(bbox[3] + padding, page_rect.height)
    )
    
    # Render the region to a pixmap
    pix = page.get_pixmap(clip=padded_rect, matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
    img_bytes = pix.tobytes("png")
    
    # Convert to base64 for embedding
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return {
        "image_data": img_base64,
        "image_url": f"data:image/png;base64,{img_base64}",
        "dimensions": {"width": pix.width, "height": pix.height}
    }

@app.post("/extract-particulars")
async def extract_particulars(file: UploadFile):
    """
    Extract particulars data from PDF with snippet images for Google Sheets.
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
                    # Create snippet image for this row
                    snippet = create_snippet_image(page, row["full_row_bbox"])
                    
                    particulars_entry = {
                        "page": page_num + 1,
                        "s_number": row["s_number"],
                        "particulars": row["particulars"],
                        "qty": row["qty"],
                        "snippet_image": snippet["image_url"],
                        "snippet_base64": snippet["image_data"],
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
            "google_sheets_ready": True
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/extract-particulars-for-sheets")
async def extract_particulars_for_sheets(file: UploadFile):
    """
    Extract particulars data formatted specifically for Google Sheets integration.
    Returns data in a format that n8n can easily process.
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
                    # Create snippet image
                    snippet = create_snippet_image(page, row["full_row_bbox"])
                    
                    # Format for Google Sheets
                    sheets_row = {
                        "S_Number": row["s_number"],
                        "Particulars": row["particulars"],
                        "Quantity": row["qty"],
                        "Page": page_num + 1,
                        "Image_Note": f"=IMAGE(\"{snippet['image_url']}\")",  # Google Sheets IMAGE formula
                        "Snippet_URL": snippet["image_url"],
                        "Hover_Text": f"Product: {row['particulars']} | Qty: {row['qty']} | Page: {page_num + 1}"
                    }
                    
                    sheets_data.append(sheets_row)
        
        doc.close()
        
        return {
            "success": True,
            "sheets_data": sheets_data,
            "total_rows": len(sheets_data),
            "columns": ["S_Number", "Particulars", "Quantity", "Page", "Image_Note", "Snippet_URL", "Hover_Text"]
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
    padding: int = Form(20)
):
    """
    Get snippet image for specific coordinates (for testing/debugging).
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
        snippet = create_snippet_image(selected_page, bbox, padding)
        
        # Return as streaming image
        img_bytes = base64.b64decode(snippet["image_data"])
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )