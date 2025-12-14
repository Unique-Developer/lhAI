from pydantic import BaseModel, Field
from typing import Optional

# This model defines the optional metadata fields from the admin upload (Requirement 1)
class ProductMetadata(BaseModel):
    sku: str = Field(..., description="Unique product identifier (e.g., 'A101-BLK')")
    material: Optional[str] = Field(None, description="Material (e.g., 'Leather', 'Cotton')")
    size: Optional[str] = Field(None, description="Admin-entered size (e.g., 'L', '42cm')")
    price: Optional[float] = Field(None, description="Product price (e.g., 49.99)")
    notes: Optional[str] = Field(None, description="Any additional notes or description")
    # File is handled separately in the FastAPI route, not in this JSON body