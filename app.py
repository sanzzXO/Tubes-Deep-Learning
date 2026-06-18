import streamlit as st
import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


st.set_page_config(
    page_title="Vehicle Door Detection",
    page_icon="🚗",
    layout="centered"
)

LABELS = [
    "Front Left Door",
    "Front Right Door",
    "Rear Left Door",
    "Rear Right Door",
    "Hood"
]


@st.cache_resource
def load_model():

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        5
    )

    model.load_state_dict(
        torch.load(
            "best_resnet18.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        logits = model(image)

        probs = torch.sigmoid(logits)

        preds = (
            probs > 0.5
        ).int()

    return (
        probs.squeeze().tolist(),
        preds.squeeze().tolist()
    )


st.title("🚗 Vehicle Component Detection")

st.write(
    """
    Upload gambar mobil untuk memprediksi kondisi:
    
    - Front Left Door
    - Front Right Door
    - Rear Left Door
    - Rear Right Door
    - Hood
    """
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    probs, preds = predict(image)

    st.subheader("Prediction")

    binary_output = []

    for label, pred, prob in zip(
        LABELS,
        preds,
        probs
    ):

        binary_output.append(pred)

        if pred == 1:
            st.success(
                f"{label}: OPEN ({prob:.2%})"
            )
        else:
            st.error(
                f"{label}: CLOSED ({prob:.2%})"
            )

    st.subheader("Binary Output")

    st.code(binary_output)

    st.subheader("Detected Open Components")

    opened = [
        LABELS[i]
        for i,v in enumerate(preds)
        if v == 1
    ]

    if len(opened) == 0:
        st.write("All components are closed.")
    else:
        for item in opened:
            st.write(f"{item}")