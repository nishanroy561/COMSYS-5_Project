import torch
from torchvision import transforms
from sklearn.metrics import classification_report
from model import GenderClassifierResNet18
from load_data import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
DATA_DIR = "processed_data"
BATCH_SIZE = 8

# Load data
_, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
print("Classes:", class_names)

# Load model
model = GenderClassifierResNet18().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Predict
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
