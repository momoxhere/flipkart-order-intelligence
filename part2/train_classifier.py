# part2/train_classifier.py
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Setup directories
os.makedirs("part2/results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Export Sample PNGs (Before transforms)
# ==========================================
print("\n--- Exporting Sample Images ---")
raw_test = datasets.FashionMNIST(root='./data', train=False, download=True)
classes = raw_test.classes

targets_needed = {
    "T-shirt/top": "00_tshirt_top.png",
    "Trouser": "01_trouser.png",
    "Sandal": "05_sandal.png",
    "Sneaker": "07_sneaker.png",
    "Ankle boot": "09_ankle_boot.png"
}

found = set()
for idx, (img, label) in enumerate(raw_test):
    class_name = classes[label]
    if class_name in targets_needed and class_name not in found:
        filename = targets_needed[class_name]
        img.save(f"data/sample_images/{filename}")
        found.add(class_name)
    if len(found) == len(targets_needed):
        break
print("Saved 5 sample PNGs to data/sample_images/")

# ==========================================
# 2. Preprocessing & Data Loading
# ==========================================
print("\n--- Preparing Data ---")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 55k train, 5k validation split
train_data, val_data = torch.utils.data.random_split(
    full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
)

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ==========================================
# 3. Transfer Learning & Feature Caching
# ==========================================
print("\n--- Initializing ResNet-18 ---")
backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = backbone.fc.in_features
backbone.fc = nn.Identity() # Remove the final layer to extract features
backbone = backbone.to(device)
backbone.eval()

def cache_features(loader, desc=""):
    print(f"Caching features for {desc} (this takes a few minutes on CPU)...")
    features_list, labels_list = [], []
    start_time = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            outputs = backbone(images)
            features_list.append(outputs.cpu())
            labels_list.append(labels)
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(loader)} batches...")
    
    print(f"Finished {desc} caching in {time.time()-start_time:.1f}s")
    return torch.cat(features_list), torch.cat(labels_list)

train_features, train_labels = cache_features(train_loader, "Training Data")
val_features, val_labels = cache_features(val_loader, "Validation Data")

# Create new dataloaders from cached features
cached_train = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
cached_val = DataLoader(TensorDataset(val_features, val_labels), batch_size=batch_size, shuffle=False)

# ==========================================
# 4. Train the Classifier Head
# ==========================================
print("\n--- Training Classifier Head ---")
head = nn.Linear(num_ftrs, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    head.train()
    running_loss = 0.0
    for features, labels in cached_train:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = head(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in cached_val:
            features, labels = features.to(device), labels.to(device)
            outputs = head(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(cached_train):.4f} | Val Accuracy: {val_acc:.2f}%")

if val_acc > 80.0:
    print("Validation accuracy exceeded 80%. Fine-tuning deeper layers is not required.")

# ==========================================
# 5. Final Test Evaluation
# ==========================================
print("\n--- Evaluating on Untouched Test Set ---")
# Cache test features once for evaluation
test_features, test_labels = cache_features(test_loader, "Test Data")

head.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for features, labels in DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size):
        features = features.to(device)
        outputs = head(features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Generate reports
test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv("part2/results/confusion_matrix.csv")
print("\nConfusion Matrix saved to part2/results/confusion_matrix.csv")

report = classification_report(all_labels, all_preds, target_names=classes)
with open("part2/results/classification_report.txt", "w") as f:
    f.write(report)
print("Classification Report saved to part2/results/classification_report.txt")

# ==========================================
# 6. Save Complete Reconstructed Model
# ==========================================
print("\n--- Saving Final Model ---")
# Attach the trained head back onto the backbone
backbone.fc = head
torch.save(backbone.state_dict(), "models/product_classifier.pt")
print("Full model successfully saved to models/product_classifier.pt")