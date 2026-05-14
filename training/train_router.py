import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_router():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Paths
    BASE_DIR = "/Users/recepozturk/Desktop/uveitis_project"
    DATA_DIR = os.path.join(BASE_DIR, "data_work", "router_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "models")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Model: MobileNetV3-Small for ultra-fast inference
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # Save class mapping
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        train_acc = train_correct.float() / len(train_dataset)
        val_acc = val_correct.float() / len(val_dataset)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_dataset):.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_dataset):.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, "router_mobilenet_best.pth")
            
            # Save checkpoint with class names
            checkpoint = {
                'state_dict': model.state_dict(),
                'classes': class_names
            }
            torch.save(checkpoint, save_path)
            print(f"Saved new best model with acc {best_acc:.4f}")

    print("Training finished!")

if __name__ == "__main__":
    train_router()
