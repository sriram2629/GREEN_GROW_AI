import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import logging

# Set up logging to provide clear information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MODEL DEFINITION (Unchanged) ---
# The model architecture is sound for this task.
class WasteClassifierCNN(nn.Module):
    def __init__(self):
        super(WasteClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary: 2 output classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- DATASET HANDLING (Unchanged) ---
# Your custom dataset class to gracefully handle corrupted images is a good approach.
class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # Using torchvision.datasets.ImageFolder to find images and labels
        self.dataset = datasets.ImageFolder(root=root, transform=None)
        self.transform = transform
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            # Load image with PIL
            img = Image.open(path).convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except Exception as e:
            logger.error(f"Skipping corrupted image {path}: {str(e)}")
            # Return a placeholder for corrupted images
            return None, -1

# --- DATASET PREPARATION (Improved) ---
def prepare_dataset():
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_full_dataset = CustomImageFolder(root='dataset/train', transform=transform)
    test_full_dataset = CustomImageFolder(root='dataset/test', transform=transform)
    
    # --- CRITICAL DEBUGGING STEP ---
    # Verify that the folder names are mapped to the correct class indices
    class_to_idx = train_full_dataset.dataset.class_to_idx
    logger.info(f"ImageFolder found the following class-to-index mapping: {class_to_idx}")
    if class_to_idx.get('Compostable') != 0 or class_to_idx.get('Non-Compostable') != 1:
        logger.warning("WARNING: Class mapping does not match app.py expectations!")
        logger.warning("Expected: {'Compostable': 0, 'Non-Compostable': 1}")
        logger.warning("Please rename your dataset folders to 'Compostable' and 'Non-Compostable'.")
    else:
        logger.info("Class mapping is correct. Proceeding with training.")

    # Filter out corrupted images that were marked with target == -1
    train_valid_indices = [i for i, (path, target) in enumerate(train_full_dataset.samples) if target != -1]
    test_valid_indices = [i for i, (path, target) in enumerate(test_full_dataset.samples) if target != -1]

    if not train_valid_indices:
        logger.error("No valid images found in the training dataset. Please check the 'dataset/train' folder.")
        return None, None
    if not test_valid_indices:
        logger.error("No valid images found in the testing dataset. Please check the 'dataset/test' folder.")
        return None, None
        
    train_dataset = Subset(train_full_dataset, train_valid_indices)
    test_dataset = Subset(test_full_dataset, test_valid_indices)
    
    # Custom collate function to filter out None values from the batch
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader

# --- TRAINING & TESTING FUNCTION (Improved) ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = WasteClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, test_loader = prepare_dataset()
    if train_loader is None or test_loader is None:
        return # Stop if datasets could not be loaded
        
    epochs = 10
    logger.info("Starting model training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # The loader now only provides valid images, so no inner checks are needed
        for images, labels in train_loader:
            if images is None or labels is None: continue # Skip batch if collate_fn made it empty
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.2f}%")
    
    logger.info("Training finished.")
    torch.save(model.state_dict(), "waste_classifier.pth")
    logger.info("Model saved successfully as waste_classifier.pth")
    
    # --- Testing phase ---
    logger.info("Starting model evaluation on the test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if images is None or labels is None: continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    test_acc = (100 * test_correct / test_total) if test_total > 0 else 0
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    # Ensure dataset directories exist before starting
    train_path = os.path.join("dataset", "train")
    test_path = os.path.join("dataset", "test")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Dataset directory not found.")
        logger.error("Please create 'dataset/train' and 'dataset/test' folders.")
        exit()
    if not os.path.exists(os.path.join(train_path, "Compostable")) or \
       not os.path.exists(os.path.join(train_path, "Non-Compostable")):
        logger.error("Class folders 'Compostable' and 'Non-Compostable' must exist in 'dataset/train/'.")
        exit()
        
    train_model()