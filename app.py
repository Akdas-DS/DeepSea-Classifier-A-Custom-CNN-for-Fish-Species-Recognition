import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------
# Class names from dataset
# -----------------
CLASS_NAMES = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]
NUM_CLASSES = len(CLASS_NAMES)

# -----------------
# Define model architecture
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
# Load trained model
# -----------------
MODEL_PATH = "fish_cnn_model.pth"  # Put your model in same folder as app.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishCNN(NUM_CLASSES)

# Load with flexibility for mismatched classes
state_dict = torch.load(MODEL_PATH, map_location=device)
model_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# -----------------
# Image transforms
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

    st.success(f"**Predicted Fish Species:** {CLASS_NAMES[predicted.item()]} 🐟")
    st.info(f"**Confidence:** {confidence:.2f}%")
