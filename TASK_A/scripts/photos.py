import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import GenderClassifierResNet18
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
IMAGES_DIR = "views"
BATCH_SIZE = 8

# Load modelS
model = GenderClassifierResNet18().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Image transform (match training pipeline)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore

# Bias correction: adjust threshold to compensate for model bias
# Based on the debug results, the model is biased towards male
# We'll use a higher threshold for male prediction
MALE_THRESHOLD = 0.87  # Require 87% confidence for male prediction

def get_class_mapping():
    """Get the class mapping from the training data structure"""
    train_dir = "data/train"
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        print(f"📋 Class mapping: {classes}")
        return classes
    else:
        # Fallback to alphabetical order
        return ["female", "male"]

def predict_gender(image_path, class_names):
    """Predict gender for a single image with bias correction"""
    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot load image: {image_path}")
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print(f"❌ No face detected in: {image_path}")
            return None
            
        # Process first detected face
        (x, y, w, h) = faces[0]
        roi_color = frame[y:y+h, x:x+w]
        img_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Transform and predict
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)  # type: ignore
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        female_prob = probabilities[0][0].item()
        male_prob = probabilities[0][1].item()
        
        # Apply bias correction: require higher confidence for male prediction
        if male_prob >= MALE_THRESHOLD:
            gender = "male"
            confidence = male_prob
        else:
            gender = "female"
            confidence = female_prob
            
        return gender, confidence, female_prob, male_prob
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def main():
    print("🔍 Testing images in 'views' folder with bias correction...")
    print(f"🎯 Male threshold: {MALE_THRESHOLD:.1%}")
    print("=" * 60)
    
    # Get class mapping
    class_names = get_class_mapping()
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Directory '{IMAGES_DIR}' not found!")
        return
        
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No image files found in '{IMAGES_DIR}'!")
        return
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(IMAGES_DIR, image_file)
        print(f"\n📷 Processing: {image_file}")
        
        result = predict_gender(image_path, class_names)
        if result:
            gender, confidence, female_prob, male_prob = result
            print(f"   ✅ Prediction: {gender.upper()} (confidence: {confidence:.2%})")
            print(f"   📊 Raw probs - Female: {female_prob:.2%}, Male: {male_prob:.2%}")
            results.append((image_file, gender, confidence))
        else:
            print(f"   ❌ Failed to process")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY (with bias correction):")
    print("=" * 60)
    
    if results:
        male_count = sum(1 for _, gender, _ in results if gender == "male")
        female_count = sum(1 for _, gender, _ in results if gender == "female")
        
        print(f"Total images processed: {len(results)}")
        print(f"Male predictions: {male_count}")
        print(f"Female predictions: {female_count}")
        
        print("\nDetailed results:")
        for image_file, gender, confidence in results:
            print(f"  {image_file}: {gender.upper()} ({confidence:.2%})")
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    main() 