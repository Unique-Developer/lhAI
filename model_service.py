import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Configuration (Global variables for efficiency)
# Use a smaller, faster CLIP model for initial testing
MODEL_NAME = "openai/clip-vit-base-patch32"

# Global variables to store the model and processor once loaded
MODEL = None
PROCESSOR = None

def load_clip_model():
    """Loads the CLIP model and processor, running on GPU if available."""
    global MODEL, PROCESSOR
    if MODEL is None:
        print(f"Loading CLIP model: {MODEL_NAME}...")
        
        # Check for GPU (CUDA)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model and processor
        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        model.eval() # <-- Add this line
        PROCESSOR = CLIPProcessor.from_pretrained(MODEL_NAME)
        
        print(f"CLIP model loaded on device: {device}")
    return MODEL, PROCESSOR, device

def get_image_embedding(image_path: str, model, processor, device) -> list[float]:
    """Generates a vector embedding for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image and move to the device
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate the embedding
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'].to(device))
            # Normalize and convert to list (CRITICAL STEP)
            image_vector = image_features.cpu().numpy().flatten().tolist()
        return image_vector
        
    except Exception as e:
        print(f"Error generating image embedding for {image_path}: {e}")
        return []

# Placeholder for text embedding (will be used for search)
def get_text_embedding(text: str, model, processor, device) -> list[float]:
    """Generates a vector embedding for a single text string."""
    try:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs.to(device))
            # Normalize and convert to list (CRITICAL STEP)
            text_vector = text_features.cpu().numpy().flatten().tolist()
        return text_vector
    except Exception as e:
        print(f"Error generating text embedding for '{text}': {e}")
        return []

if __name__ == '__main__':
    # Test loading the model
    model, processor, device = load_clip_model()
    # The dimension of the CLIP ViT-B/32 embedding is 512
    print(f"Model Feature Dimension: {model.config.projection_dim}")