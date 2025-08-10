import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# -----------------
# Load class names automatically
# -----------------
train_dir = r"C:\Users\makda\OneDrive\Desktop\Project_3\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train"
class_names = sorted(os.listdir(train_dir))  # Reads folder names as classes
num_classes = len(class_names)

# -----------------
# Define model (must match your training architecture)
# -----------------
class FishCNN(nn.Module):
    def __init__(self, num_classes):
        super(FishCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -----------------
# Load model
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishCNN(num_classes)
model.load_state_dict(torch.load(r"C:\Users\makda\OneDrive\Desktop\Project_3\fish_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------
# Define transforms
# -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------
# Streamlit UI
# -----------------
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish to identify its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100

    st.success(f"**Predicted Fish Species:** {class_names[predicted.item()]}")
    st.info(f"**Confidence:** {confidence:.2f}%")
