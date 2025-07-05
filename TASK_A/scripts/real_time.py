import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import GenderClassifierResNet18

# Load labels and device
class_names = ['female', 'male']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = GenderClassifierResNet18().to(device)
model.load_state_dict(torch.load("TASK_A/best_model.pth", map_location=device))
model.eval()

# 👇 Use the same transform as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # match training
                         std=[0.229, 0.224, 0.225])
])

# Setup face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore
cap = cv2.VideoCapture(0)

print("🎥 Starting real-time gender detection...")
print("💡 Press 'q' to quit the application")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        img_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        transformed_img = transform(pil_img)  # type: ignore
        input_tensor = transformed_img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            pred_idx = int(pred)
            if 0 <= pred_idx < len(class_names):
                label = class_names[pred_idx]
            else:
                label = "unknown"

        # 🔲 Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    # Add quit instruction to the frame
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    
    cv2.imshow("Gender Prediction - Press 'q' to quit", frame)
    if cv2.waitKey(1) == ord("q"):
        print("👋 Quitting application...")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Application closed successfully")
