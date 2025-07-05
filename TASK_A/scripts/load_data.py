import os
import shutil
from PIL import Image
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

RAW_DATA_DIR = "data"
PROCESSED_DATA_DIR = "processed_data"
IMAGE_SIZE = 224

# Common transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def balance_and_process_images(src_dir, dst_dir):
    class_counts = {}

    # Count images per class
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        image_files = os.listdir(class_path)
        class_counts[class_name] = image_files

    # Determine minority class count
    min_count = min(len(imgs) for imgs in class_counts.values())

    for class_name, image_files in class_counts.items():
        random.shuffle(image_files)
        selected_images = image_files[:min_count]  # Downsample to balance
        dst_class_path = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        for img_name in selected_images:
            img_path = os.path.join(src_dir, class_name, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                img_pil = transforms.ToPILImage()(img)
                save_path = os.path.join(dst_class_path, img_name)
                img_pil.save(save_path)
            except Exception as e:
                print(f"[ERROR] Skipping {img_path} due to {e}")

# 👉 This is the function used in train.py
def get_data_loaders(data_dir="processed_data", batch_size=8):
    transform_loader = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_loader)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_loader)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("🔁 Balancing and processing training images...")
    balance_and_process_images(os.path.join(RAW_DATA_DIR, "train"), os.path.join(PROCESSED_DATA_DIR, "train"))

    print("🔁 Balancing and processing validation images...")
    balance_and_process_images(os.path.join(RAW_DATA_DIR, "val"), os.path.join(PROCESSED_DATA_DIR, "val"))

    print("✅ Dataset balanced and saved to 'processed_data/'")
