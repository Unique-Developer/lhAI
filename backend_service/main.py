import uvicorn
import shutil
import os
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
import torch 

from .qdrant_client import get_product_point

# --- Import Dependent Modules ---
from .models import ProductMetadata
from .database import (
    get_db_connection, 
    insert_product_metadata,
    get_unprocessed_products, 
    mark_product_as_searchable 
)
from .ingestion_worker import process_product_data
from .model_service import load_clip_model, get_image_embedding, get_text_embedding 
from .qdrant_client import setup_qdrant_collection, upsert_vector, search_vectors 


# Load environment variables from the project root .env file
load_dotenv()

# --- Configuration ---
UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER", "../uploads"))

# Ensure the uploads directory exists
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL ERROR: Could not create upload directory {UPLOAD_DIR}: {e}")
    sys.exit(1) 

app = FastAPI(
    title="Multimodal Catalog Retrieval API",
    description="API for admin to upload product images and for users to perform multimodal search."
)

# Global variables for the Multimodal Model
MODEL, PROCESSOR, DEVICE = None, None, None


@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant collection and load the Multimodal Model."""
    global MODEL, PROCESSOR, DEVICE
    
    print("--- Starting Multimodal Service Initialization ---")
    
    # 1. Setup Qdrant
    if not setup_qdrant_collection():
         print("WARNING: Qdrant setup failed. Vector search will be unavailable.")

    # 2. Load Model
    MODEL, PROCESSOR, DEVICE = load_clip_model()
    if MODEL is None:
         print("CRITICAL: CLIP Model failed to load. Embedding generation will fail.")
         
    print("--- Initialization Complete ---")


@app.get("/")
def read_root():
    return {"message": "Multimodal Catalog Retrieval Service is Running"}

@app.get("/debug/qdrant/{product_id}", tags=["Debug"])
def debug_qdrant_point(product_id: int):
    """TEMPORARY: Retrieves the raw payload stored in Qdrant for a product ID."""
    payload = get_product_point(product_id)
    if not payload:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Product ID not found in Qdrant.")
    return {"id": product_id, "payload": payload}


@app.post("/admin/upload/product", status_code=status.HTTP_202_ACCEPTED)
async def upload_product(
    file: UploadFile = File(..., description="Product Image or PDF catalog file"),
    sku: str = Form(..., description="Unique Product Identifier"),
    material: Optional[str] = Form(None),
    size_admin: Optional[str] = Form(None, alias="size_admin"),
    price: Optional[float] = Form(None),
    notes: Optional[str] = Form(None),
):
    """
    Handles file upload, saves the file, and inserts initial metadata into PostgreSQL, 
    then initiates OCR/Extraction.
    """
    
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in ['.png', '.jpg', '.jpeg', '.pdf']:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PNG, JPG, JPEG, and PDF are supported."
        )

    save_filename = f"{sku}_{Path(file.filename).stem}{file_extension}"
    file_location = UPLOAD_DIR / save_filename
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        product_metadata = {
            "sku": sku,
            "material": material,
            "size_admin": size_admin, 
            "price": price,
            "notes": notes,
            "image_path": str(file_location.resolve()) 
        }
        
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed during ingestion initiation."
            )
            
        product_id = insert_product_metadata(conn, product_metadata)
        conn.close()

        # Initiate the OCR/Extraction Worker
        process_product_data(product_id, str(file_location.resolve()), sku)

        return {
            "message": "Product uploaded and OCR/Extraction initiated. Ready for vector indexing.",
            "product_id": product_id,
            "sku": sku
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        if file_location.exists():
             os.remove(file_location)
             
        error_detail = str(e).splitlines()[0]
        if 'duplicate key value violates unique constraint "products_sku_key"' in error_detail:
             raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"SKU '{sku}' already exists. Use a unique SKU."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {error_detail}"
        )


@app.post("/admin/ingest/vectors", status_code=status.HTTP_200_OK)
async def ingest_vectors(limit: int = 50):
    """
    Triggers the incremental vector ingestion pipeline.
    Reads unprocessed products, generates embeddings, and upserts to Qdrant.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal Model Service is not initialized."
        )

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed."
        )
        
    products = get_unprocessed_products(conn, limit)
    processed_count = 0
    
    for product in products:
        product_id = product['id']
        image_path = product['image_path']
        
        # 1. Generate Image Embedding
        image_vector = get_image_embedding(image_path, MODEL, PROCESSOR, DEVICE)
        
        if not image_vector:
            print(f"Skipping ID {product_id}: Image vector generation failed.")
            continue
            
        # 2. Prepare Payload (Metadata for Filtering and Search Text)
        price_val = float(product['price']) if product['price'] is not None else 0.0
        material_str = product['material'] or "unknown"
        
        # Ensure material is uppercase for filtering consistency
        material_upper = material_str.upper() 
        
        # Pull extracted text fields for search boosting
        item_no = product.get('item_no_ocr', '') or ''
        size_extracted = product.get('size_extracted', '') or ''
        
        payload = {
            "sku": product['sku'],
            "material": material_upper, # Primary material field
            "price": price_val,
            "size_admin": product['size_admin'] or "unknown",
            
            # --- Search Relevance Enhancement: Structured Search Text ---
            "search_text": (
                f"Item No: {item_no}. Size: {size_extracted}. "
                f"Material: {material_upper}. Price: {price_val}. "
                f"Notes: {product['notes'] or ''}. "
                f"OCR Text: {product['extracted_text'] or ''}"
            )
        }
        
        # 3. Upsert to Qdrant
        if upsert_vector(product_id, image_vector, payload):
            # 4. Mark as processed in PostgreSQL
            mark_product_as_searchable(conn, product_id)
            processed_count += 1
            
    conn.close()
    
    return {
        "message": f"Successfully processed and indexed {processed_count} products into Qdrant.",
        "total_unprocessed": len(products) - processed_count
    }


@app.get("/search/text", status_code=status.HTTP_200_OK)
async def search_by_text(
    query: str,
    top_k: int = 10,
    material: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
):
    """
    Performs a multimodal search using a text query (e.g., "tall grey glass chandelier").
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal Model Service is not initialized."
        )

    # 1. Convert Text Query to Vector Embedding
    query_vector = get_text_embedding(query, MODEL, PROCESSOR, DEVICE)
    
    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate vector for the search query."
        )

    # 2. Prepare Price Filter Tuple
    price_range = None
    if min_price is not None and max_price is not None and min_price <= max_price:
        price_range = (min_price, max_price)
        
    # 3. Prepare Material Filter (Uppercase for consistency)
    material_filter = material.upper() if material else None

    # 4. Search Qdrant
    results = search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        material_filter=material_filter,
        price_range=price_range
    )
    
    if not results:
        return {"message": "No products found matching the query and filters.", "results": []}

    return {
        "message": f"Found {len(results)} relevant products.",
        "query": query,
        "results": results
    }


# Main entry point to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)