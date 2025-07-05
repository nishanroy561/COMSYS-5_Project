import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Paths
VAL_DIR = 'TASK_B/val'
EMBEDDING_PATH = 'TASK_B/train_embeddings.pt'

# Load stored embeddings from train
person_embeddings = torch.load(EMBEDDING_PATH, map_location=device)

# Transform for images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to get embedding
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device) # type: ignore
        with torch.no_grad():
            emb = model(img)
        return emb.squeeze()
    except Exception as e:
        print(f"[ERROR] Skipping {img_path}: {e}")
        return None

# Similarity threshold
THRESHOLD = 0.7

# Metrics
tp, fp, tn, fn = 0, 0, 0, 0

# Evaluation loop
for person_folder in tqdm(os.listdir(VAL_DIR), desc="Evaluating"):
    val_person_path = os.path.join(VAL_DIR, person_folder)
    
    if not os.path.isdir(val_person_path):
        continue

    images = [f for f in os.listdir(val_person_path) if f.endswith('.jpg')]

    # Include distorted images if any
    distort_dir = os.path.join(val_person_path, 'distortion')
    if os.path.exists(distort_dir):
        images += [os.path.join('distortion', f) for f in os.listdir(distort_dir) if f.endswith('.jpg')]

    for img_name in images:
        img_path = os.path.join(val_person_path, img_name)
        emb = get_embedding(img_path)
        if emb is None: continue

        matched = False
        for ref_person, ref_embeddings in person_embeddings.items():
            for ref_emb in ref_embeddings:
                score = cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
                if score >= THRESHOLD:
                    if ref_person == person_folder:
                        tp += 1
                    else:
                        fp += 1
                    matched = True
                    break
            if matched:
                break

        if not matched:
            if person_folder in person_embeddings:
                fn += 1  # should've matched but didn't
            else:
                tn += 1  # no match expected and none found

# Metrics calculation
def safe_div(x, y):
    return x / y if y > 0 else 0

accuracy = safe_div(tp + tn, tp + tn + fp + fn)
precision = safe_div(tp, tp + fp)
recall = safe_div(tp, tp + fn)
f1 = safe_div(2 * precision * recall, precision + recall)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
