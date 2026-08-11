# part2/predict_image.py
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

def classify_product_image(image_path: str) -> dict:
    """
    Loads the saved model, preprocesses a real PNG file, runs inference,
    and returns the predicted category and confidence score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Reconstruct Model architecture
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    # 2. Load Weights
    model.load_state_dict(torch.load("models/product_classifier.pt", map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. Preprocessing (Identical to training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Inference
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return {
        "predicted_category": classes[predicted_idx.item()],
        "confidence": round(confidence.item(), 4)
    }

# Quick test if script is run directly
if __name__ == "__main__":
    result = classify_product_image("data/sample_images/09_ankle_boot.png")
    print(f"Test Classification: {result}")