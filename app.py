import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import urllib.request

st.title("Image Classifier")

# Load model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Load labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    label = classes[pred.item()]
    st.write(f"Prediction: {label}")