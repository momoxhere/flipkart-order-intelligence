# part3/tools.py
import joblib
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image

# ==========================================
# Tool 1: Return Risk
# ==========================================
def check_return_risk(order_features: dict) -> dict:
    """Loads Part 1 Random Forest and predicts return risk bucket."""
    try:
        model = joblib.load("models/return_risk_model.pkl")
    except FileNotFoundError:
        return {"error": "Return risk model artifact not found."}
        
    df = pd.DataFrame([order_features])
    prob = model.predict_proba(df)[0][1]
    
    # Risk bucket calibration anchored to t*_rf
    t_star_rf = 0.36 
    
    if prob < t_star_rf:
        bucket = "Low"
    elif t_star_rf <= prob < (t_star_rf + 0.15):
        bucket = "Medium"
    else:
        bucket = "High"
        
    return {
        "return_probability": round(prob, 4),
        "risk_bucket": bucket,
        "t_star_rf_used": t_star_rf
    }

# ==========================================
# Tool 2: Product Image Classification
# ==========================================
def classify_product_image(image_path: str) -> dict:
    """Loads Part 2 PyTorch model and predicts apparel category."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load("models/product_classifier.pt", map_location=device))
        model = model.to(device)
        model.eval()
    except FileNotFoundError:
        return {"error": "Product classifier artifact not found."}

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return {"error": f"Image file not found at {image_path}"}
        
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return {
        "predicted_category": classes[predicted_idx.item()],
        "confidence": round(confidence.item(), 4)
    }