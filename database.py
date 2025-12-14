import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def insert_product_metadata(conn, product_data: dict):
    """Inserts product metadata into the 'products' table."""
    cursor = conn.cursor()
    
    # Check for 'size' logic (Requirement 1)
    # If size is not provided, treat it as 'unknown' in the size_admin field.
    size_admin = product_data.get('size', 'unknown')
    
    # The SQL query inserts the required fields. 
    # NOTE: size_ocr and extracted_text are left as NULL for now, 
    # they will be updated later in the Ingestion Pipeline.
    sql = """
    INSERT INTO products (sku, material, size_admin, price, notes, image_path)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    
    try:
        cursor.execute(sql, (
            product_data['sku'], 
            product_data.get('material'), 
            size_admin, 
            product_data.get('price'), 
            product_data.get('notes'), 
            product_data['image_path'] # This path is returned after file save
        ))
        product_id = cursor.fetchone()[0]
        conn.commit()
        return product_id
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Database insertion error: {e}")
        raise e
    finally:
        cursor.close()

    # In backend_service/database.py, add these functions:

def get_unprocessed_products(conn, limit=100):
    """Fetches products that have been processed by OCR but not yet indexed in Qdrant."""
    cursor = conn.cursor()
    # Check for is_searchable = FALSE AND extracted_text IS NOT NULL (meaning OCR ran)
    sql = """
    SELECT 
        id, sku, material, price, size_admin, size_extracted, notes, extracted_text, image_path
    FROM products
    WHERE is_searchable = FALSE AND extracted_text IS NOT NULL
    LIMIT %s;
    """
    cursor.execute(sql, (limit,))
    products = cursor.fetchall()
    cursor.close()
    
    # Return as a list of dictionaries for easier processing
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in products]

def mark_product_as_searchable(conn, product_id: int):
    """Sets the is_searchable flag to TRUE after successful Qdrant ingestion."""
    cursor = conn.cursor()
    sql = """
    UPDATE products
    SET is_searchable = TRUE
    WHERE id = %s;
    """
    cursor.execute(sql, (product_id,))
    conn.commit()
    cursor.close()