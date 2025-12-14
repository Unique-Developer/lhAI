import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models 
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, Range
from typing import Optional, Tuple, List # Added necessary type hints

# Load environment variables
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", 512))
TIMEOUT = 10 

# Initialize the Qdrant Client (Global Object)
try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=TIMEOUT)
except Exception as e:
    print(f"Error initializing Qdrant client: {e}")
    client = None

def setup_qdrant_collection() -> bool:
    """
    Ensures the vector collection exists, forcing recreation if necessary, 
    and sets up the payload indexes, including 'material' as a keyword index.
    """
    if client is None:
        print("Qdrant client is not available for setup.")
        return False
        
    # Check if collection already exists
    try:
        collections = client.get_collections().collections
        if COLLECTION_NAME in [c.name for c in collections]:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            # We don't recreate the whole collection if it exists, but we must 
            # ensure indices are correct, which is why manual recreation may be needed.
            # We proceed to explicitly create indices below if they are missing.
    except Exception as e:
        print(f"Failed to check collections (Qdrant likely not running): {e}")
        return False

    try:
        # 1. Define the OptimizersConfig
        optimizer_config_model = models.OptimizersConfig(
            default_segment_number=2,
            deleted_threshold=0.6,
            vacuum_min_vector_number=1000,
            flush_interval_sec=5
        )

        # 2. Create the collection (recreates if exists, ensuring structure)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            optimizers_config=optimizer_config_model.model_dump(), 
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
        
        # 3. Add Payload Indexes separately (Ensures proper typing for filtering)
        
        # Index material as a KEYWORD field for exact matching (e.g., 'ACRYLIC+MATEL')
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="material",
            field_schema="keyword" 
        )
        # Index price for range filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="price",
            field_schema="float" 
        )
        print(f"Payload indexes created for 'material' (keyword) and 'price' (float).")

        return True
    except Exception as e:
        print(f"Error creating Qdrant collection or indexes: {e}")
        return False

def upsert_vector(
    product_id: int, 
    vector: List[float], 
    payload: dict
) -> bool:
    """Upserts a single vector and its metadata payload to the collection."""
    global client
    if client is None:
        print(f"Error upserting vector for ID {product_id}: Qdrant client is None.")
        return False

    try:
        operation_info = client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                models.PointStruct(
                    id=product_id, 
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        # --- Check Qdrant Status Explicitly ---
        if operation_info.status == models.UpdateStatus.COMPLETED:
            print(f"SUCCESS: Point {product_id} upserted. Status: COMPLETED.")
            return True
        else:
            print(f"WARNING: Point {product_id} upsert failed. Status: {operation_info.status}. Operation ID: {operation_info.operation_id}")
            return False
            
    except Exception as e:
        print(f"CRITICAL ERROR during Qdrant upsert for ID {product_id}: {e}")
        return False
# In backend_service/qdrant_client.py, add this function:

def get_product_point(product_id: int) -> Optional[dict]:
    """Retrieves a single point's payload from Qdrant by its ID."""
    if client is None:
        return None
    try:
        point = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[product_id],
            with_payload=True
        )
        if point:
            return point[0].payload
        return None
    except Exception as e:
        print(f"Error retrieving point {product_id}: {e}")
        return None
    
def search_vectors(
    query_vector: List[float], 
    top_k: int = 10,
    material_filter: Optional[str] = None,
    price_range: Optional[Tuple[float, float]] = None
) -> List[dict]:
    """
    Performs a vector search in Qdrant with optional payload filtering.
    Uses the low-level API structure and validated MatchValue arguments.
    """
    if client is None:
        print("Qdrant client is None, skipping search.")
        return []
        
    must_filters = []
    
    # Add Material Filter: Must match the exact material string (e.g., 'ACRYLIC+MATEL')
    if material_filter:
        must_filters.append(
            models.FieldCondition(
                key="material", # Filter against the 'material' field
                # Use MatchValue with the correct argument 'value' for keyword matching
                match=models.MatchValue(value=material_filter.upper()) 
            )
        )

    # Add Price Range Filter
    if price_range and len(price_range) == 2:
        min_price, max_price = price_range
        must_filters.append(
            models.FieldCondition(
                key="price",
                range=models.Range(gte=min_price, lte=max_price)
            )
        )

    # Construct the final filter object
    query_filter = models.Filter(must=must_filters) if must_filters else None

    try:
        # Execute the search using the most stable API client structure
        search_request = models.SearchRequest(
            vector=query_vector,
            filter=query_filter, 
            limit=top_k,
            with_payload=True, 
        )
        
        search_response = client.http.points_api.search_points(
            collection_name=COLLECTION_NAME,
            search_point_request=search_request
        )
        search_result = search_response.result 
        
        # Format the results into a clean list of dictionaries
        results = []
        for hit in search_result:
            payload = hit.payload
            results.append({
                "product_id": hit.id,
                "score": hit.score,
                "sku": payload.get("sku"),
                "material": payload.get("material"),
                "price": payload.get("price"),
                "search_text_snippet": payload.get("search_text")[:100] + "..." if payload.get("search_text") else None
            })
            
        return results
        
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []

if __name__ == '__main__':
    setup_qdrant_collection()