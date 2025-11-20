import os
import io
import json
import base64
import numpy as np
from PIL import Image
from datetime import datetime

# Class-specific metadata and disposal instructions
CATEGORY_METADATA = {
    "recyclable": {
        "title": "Recyclable Waste",
        "badge_class": "badge-recyclable",
        "color": "#10b981", # emerald green
        "bin_type": "Blue / Green Recycling Bin",
        "description": "Materials that can be processed and remanufactured into new products.",
        "common_items": ["Plastic bottles", "Cardboard & Paper", "Glass containers", "Metal cans", "Tetra paks"],
        "disposal_instructions": "Rinse containers to remove food residue. Flatten cardboard boxes and crush plastic bottles to save space before placing in the blue recycling bin.",
        "environmental_impact": "Recycling 1 ton of plastic saves up to 2,000 gallons of gasoline and reduces carbon emissions significantly."
    },
    "hazardous": {
        "title": "Hazardous / Non-Recyclable Waste",
        "badge_class": "badge-hazardous",
        "color": "#ef4444", # crimson red
        "bin_type": "Red / Designated Hazardous Waste Bin",
        "description": "Materials containing toxic, corrosive, reactive, or ignitable substances.",
        "common_items": ["Batteries", "Electronic waste (E-waste)", "Chemical containers", "Fluorescent tubes", "Medical waste"],
        "disposal_instructions": "DO NOT mix with standard municipal waste. Store in a safe, leak-proof container and drop off at an authorized hazardous or e-waste collection center.",
        "environmental_impact": "Improper disposal leaches heavy metals (lead, mercury, cadmium) into groundwater and contaminates soil ecosystems."
    }
}

def decode_base64_image(base64_str):
    """
    Decode a base64 string into a PIL Image and OpenCV-compatible RGB numpy array
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(image)

def encode_image_to_base64(image_np):
    """
    Convert a numpy image (RGB) to base64 data URI
    """
    image = Image.fromarray(image_np.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def format_prediction_result(category, confidence, all_probabilities=None, inference_time_ms=0.0):
    """
    Format prediction with full metadata for API and frontend display
    """
    meta = CATEGORY_METADATA.get(category.lower(), CATEGORY_METADATA["recyclable"])
    
    result = {
        "category": category,
        "title": meta["title"],
        "confidence": round(float(confidence) * 100, 2),
        "confidence_raw": float(confidence),
        "color": meta["color"],
        "badge_class": meta["badge_class"],
        "bin_type": meta["bin_type"],
        "description": meta["description"],
        "common_items": meta["common_items"],
        "disposal_instructions": meta["disposal_instructions"],
        "environmental_impact": meta["environmental_impact"],
        "inference_time_ms": round(inference_time_ms, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_probabilities": all_probabilities or {}
    }
    return result

def save_class_indices(classes, filepath="models/class_indices.json"):
    """
    Save class labels mapping to JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mapping = {i: cls for i, cls in enumerate(classes)}
    with open(filepath, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping

def load_class_indices(filepath="models/class_indices.json"):
    """
    Load class labels mapping from JSON file
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            mapping = json.load(f)
            return {int(k): v for k, v in mapping.items()}
    return {0: "hazardous", 1: "recyclable"}
