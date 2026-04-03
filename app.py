import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import urllib.request

print("App started")

st.title("Image Classifier")

@st.cache_resource
def load_model():
    print("Loading model...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    print("Loading labels...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return urllib.request.urlopen(url).read().decode("utf-8").splitlines()

model = load_model()
classes = load_labels()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MobileNet_V2_Weights.DEFAULT.meta["mean"],
        std=MobileNet_V2_Weights.DEFAULT.meta["std"]
    )
])

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    print("File uploaded")

    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image)

        img = transform(image).unsqueeze(0)

        print("Running inference...")
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)

        label = classes[pred.item()]
        print(f"Prediction: {label}")

        st.write(f"Prediction: {label}")

    except Exception as e:
        print("Error:", e)
        st.error(f"Error: {e}")