import os
import shutil
import re
from pathlib import Path
from typing import Optional, Tuple

# Image and Document Processing Libraries
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError

# Database and Utilities
from .database import get_db_connection

# Load Tesseract Path if specified in .env
tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path:
    # Set the path to the Tesseract executable for pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# --- Helper Functions for Data Enrichment ---

def run_ocr_on_image(image_path: Path) -> str:
    """Runs OCR on a single image file and returns the extracted text."""
    try:
        img = Image.open(image_path)
        # Use English language for OCR, may need to optimize resolution for quality
        text = pytesseract.image_to_string(img, lang='eng')
        return text.strip()
    except Exception as e:
        print(f"Error running OCR on {image_path}: {e}")
        return ""

def extract_price_from_text(text: str) -> Optional[float]:
    """
    Uses flexible regex patterns to find prices, prioritizing keywords and decimal structure.
    Handles Rupee (₹) and trailing slash notation (/-).
    """
    if not text:
        return None

    # --- NEW CRITICAL STEP: EXCLUDE MODEL/ITEM NUMBERS BEFORE PRICE CHECK ---
    
    # 1. First, define the cleaning patterns for Item No/SKU lines. 
    # This pattern captures the entire line starting with "Item no" or "SKU"
    item_no_pattern = r'(?:item\s*no|model|sku)\s*[:]?\s*(\S+)'
    
    # 2. Pre-clean the text: Find and remove the entire line containing the item number.
    # We use a non-greedy match to grab the item number and its context.
    cleaned_text = re.sub(item_no_pattern, '', text, flags=re.IGNORECASE)
    
    # --- Now proceed with the cleaned_text ---
    
    # 3. Normalize cleaned text for price extraction
    normalized_text = (
        cleaned_text.replace(',', '')
            .replace('€', ' ')
            .replace('£', ' ')
            .replace('$', ' ')
            .replace('₹', ' ')   
            .replace('/-', '')    
            .replace(':', ' ')    
    )
    
    # 4. Define regex patterns (PRIORITY ORDERED - Now they only see MRP 14500)
    patterns = [
        # PRIORITY 1: CONTEXT (Keywords + Number) - This will now ONLY match MRP 14500
        r'(?:mrp|price|cost|only|sale|value)\s*(\d+\.?\d{0,2})',
        
        # PRIORITY 2: DECIMAL FORMAT (X.XX)
        r'\b(\d+\.\d{2})\b', 
        
        # PRIORITY 3: Currency Remnant 
        r'\s*(\d+\.?\d{0,2})' 
    ]

    # 5. Search patterns
    for pattern in patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            try:
                # Return the first successful match
                return float(match.group(1).replace(' ', ''))
            except ValueError:
                continue
                
    return None

def extract_structured_metadata(text: str) -> dict:
    """
    Extracts structured metadata (SKU, Size, Material, etc.) from raw OCR text
    using heuristics tailored to the product catalog layout.
    """
    if not text:
        return {}

    extracted_data = {}
    
    # Define flexible regex patterns for extraction
    patterns = {
        # Pattern for Material:
        "material": r'\bM[aA]t[eE]r[iI]al\s*[:]?\s*([A-Z\s]+)',
        
        # Pattern for Size (captures numbers, x, X, MM, CM):
        "size_extracted": r'size\s*[:]?\s*([A-Z0-9xX\s]+(?:MM|CM)?)\s*',
        
        # Pattern for Finish/Color:
        "finish": r'(?:finish|color)\s*[:]?\s*([A-Z0-9\s]+)',
        
        # Pattern for Lamp/Bulb type:
        "lamp": r'(?:lamp|bulb)\s*[:]?\s*([A-Z0-9\s]+)',
        
        # Pattern for Item No/SKU (captures common separators like /):
        "item_no_ocr": r'(?:item\s*no|model|md|sku)\s*[:]?\s*([A-Z0-9\/]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Clean up the captured group (remove leading/trailing spaces, slashes, or hyphens)
            value = match.group(1).strip().strip('/').strip('-') 
            extracted_data[key] = value
            
    return extracted_data


def process_pdf_catalog(pdf_path: Path, output_dir: Path, sku_prefix: str) -> list[Tuple[str, str]]:
    """
    Extracts pages from a PDF as images, runs OCR on each, and cleans up the original PDF.
    """
    extracted_data = []
    
    # Create a sub-folder for extracted images from this PDF
    pdf_stem = pdf_path.stem
    pdf_output_dir = output_dir / f"extracted_{pdf_stem}"
    pdf_output_dir.mkdir(exist_ok=True)
    
    try:
        # NOTE: This line requires Poppler to be installed and in the system PATH.
        pages = convert_from_path(pdf_path, 300) 
        
        for i, page in enumerate(pages):
            image_filename = f"{sku_prefix}_page{i+1}.jpg"
            image_path = pdf_output_dir / image_filename
            page.save(image_path, 'JPEG')
            
            ocr_text = run_ocr_on_image(image_path)
            extracted_data.append((str(image_path.resolve()), ocr_text))
            
        # Clean up the original PDF file after successful extraction
        os.remove(pdf_path)
        
        return extracted_data
        
    except PDFPageCountError as e:
        print(f"Error reading PDF page count for {pdf_path}: {e}")
        return []
    except Exception as e:
        print(f"General error processing PDF {pdf_path}. Check Poppler installation/PATH: {e}")
        return []


# --- Main Ingestion Logic ---

def process_product_data(product_id: int, original_file_path: str, sku: str):
    """
    Handles the full processing workflow (Image/PDF -> OCR -> Data Enrichment -> DB Update).
    """
    file_path = Path(original_file_path)
    file_extension = file_path.suffix.lower()
    
    UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER", "../uploads")) 
    products_to_ingest = []
    
    # --- Step 1: Handle File Type and Run OCR ---
    if file_extension == '.pdf':
        extracted_data = process_pdf_catalog(file_path, UPLOAD_DIR, sku)
        
        if not extracted_data:
            print(f"PDF processing failed for SKU {sku}. Cannot proceed with ingestion.")
            return

        # Aggregate OCR text and link to the first extracted image
        all_ocr_text = " ".join([text for _, text in extracted_data])
        first_image_path = extracted_data[0][0]
        products_to_ingest.append((first_image_path, all_ocr_text))
        
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        ocr_text = run_ocr_on_image(file_path)
        products_to_ingest.append((original_file_path, ocr_text))
        
    else:
        print(f"Skipping unsupported file type: {file_extension}")
        return

    # --- Step 2: Data Enrichment and DB Update ---
    conn = get_db_connection()
    if not conn:
        print("Failed to get DB connection for update.")
        return

    if products_to_ingest:
        image_path, extracted_text = products_to_ingest[0]
        
        # Run all enrichment parsers
        extracted_price = extract_price_from_text(extracted_text)
        extracted_attributes = extract_structured_metadata(extracted_text)
        
        cursor = conn.cursor()
        
        try:
            # 2a. Determine Final Price (Override 0/NULL with OCR if found)
            cursor.execute("SELECT price FROM products WHERE id = %s;", (product_id,))
            current_price = cursor.fetchone()[0]
            
            if extracted_price is not None and (current_price is None or current_price == 0.0):
                final_price = extracted_price
            else:
                final_price = current_price
            
            # 2b. Determine Final Material (Override NULL with OCR if found)
            cursor.execute("SELECT material FROM products WHERE id = %s;", (product_id,))
            current_material = cursor.fetchone()[0]
            
            final_material = current_material
            if current_material is None and extracted_attributes.get('material'):
                final_material = extracted_attributes['material']

            # 2c. Execute Final Update
            sql = """
            UPDATE products 
            SET extracted_text = %s, 
                image_path = %s,
                price = %s, 
                material = %s, 
                size_extracted = %s,
                finish = %s,
                lamp = %s,
                item_no_ocr = %s,
                is_searchable = FALSE -- Remains FALSE until embedding generation (Next Phase)
            WHERE id = %s;
            """
            
            cursor.execute(sql, (
                extracted_text, 
                image_path, 
                final_price, 
                final_material, 
                extracted_attributes.get('size_extracted'),
                extracted_attributes.get('finish'),
                extracted_attributes.get('lamp'),
                extracted_attributes.get('item_no_ocr'),
                product_id
            ))
            conn.commit()
            print(f"Successfully updated product ID {product_id} with ALL extracted data.")
            
        except Exception as e:
            conn.rollback()
            print(f"Failed to update database for ID {product_id}: {e}")
        finally:
            cursor.close()
            conn.close()

    print(f"Processing complete for product ID {product_id}.")