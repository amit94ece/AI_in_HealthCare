import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import boto3
import json
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

# === SET PAGE CONFIG FIRST ===
st.set_page_config(page_title="Lung Histopathology AI (Claude 3.7 / 3.5 via Bedrock)", page_icon="🫁", layout="wide")

# === AWS CONFIGURATION (use secrets) ===
session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_REGION"]
)

bedrock = session.client(service_name="bedrock-runtime")

# === MODEL OPTIONS ===
model_options = {
    "Claude 3.7 Sonnet": "arn:aws:bedrock:us-east-1:560429778408:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}

# === MODEL LOADING (Image Classifier) ===
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("best_lung_cancer_classifier.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['lung_aca', 'lung_n', 'lung_scc']

# === IMAGE ENCODING (for Claude) ===
def encode_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=75)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

# === PREDICTION FUNCTION ===
def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
    return class_names[predicted_class.item()], confidence.item()

# === PROMPT BUILDER ===
def build_plain_text_prompt(prediction, confidence):
    readable = {
        "lung_aca": "Lung Adenocarcinoma",
        "lung_n": "Normal Lung Tissue",
        "lung_scc": "Lung Squamous Cell Carcinoma"
    }
    pred_readable = readable.get(prediction, prediction)

    prompt = f"""
You are a medical AI trained in histopathology.

The CNN model predicted this tissue as **{pred_readable}** with **{confidence:.2f}% confidence**.

Please:
- Evaluate whether this prediction seems reasonable based on typical histopathology patterns.
- Describe key features you expect to see in this type of tissue.
- Provide a short diagnostic commentary.
- Add Disclaimer that this is an AI-generated response and should not be used as a substitute for professional medical advice.
"""
    return prompt

# === CLAUDE MULTIMODAL CALL ===
def ask_claude_bedrock_multimodal(model_id, img_base64, text_prompt):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]
    }
    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

# === GRADCAM FUNCTION ===
def generate_gradcam_heatmap(img, model):
    model.eval()
    final_conv_layer = model.layer4[-1]

    gradients = []
    activations = []

    def save_gradient(module, input, output):
        gradients.append(output[0].detach())

    def save_activation(module, input, output):
        activations.append(output.detach())

    final_conv_layer.register_forward_hook(save_activation)
    final_conv_layer.register_backward_hook(save_gradient)

    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad = True

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    gradient = gradients[0]
    activation = activations[0][0]

    weights = gradient.mean(dim=(1, 2))
    cam = torch.zeros(activation.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam.detach().numpy(), 0)
    
    # === SAFE Normalization
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min != 0:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    cam = np.uint8(cam * 255)

    cam = Image.fromarray(cam).resize(img.size, Image.BILINEAR)

    return cam


# === STREAMLIT APP ===
st.title("🫁 Lung Cancer Detection + Claude AI Explanation (Bedrock Models)")

# Sidebar Settings
st.sidebar.title("🔧 Settings")
mode = st.sidebar.radio(
    "Choose Mode:",
    ("Single Model", "Side-by-Side Comparison"),
    index=0
)

if mode == "Single Model":
    selected_model_label = st.sidebar.radio(
        "Choose Claude Model:",
        list(model_options.keys()),
        index=0
    )
    selected_model_id = model_options[selected_model_label]

uploaded_file = st.file_uploader("📤 Upload a Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔎 Predict and Ask AI"):
        with st.spinner('Predicting and Consulting Claude models...'):
            # CNN Prediction
            prediction, confidence = predict_image(img)
            st.success(f"✅ Predicted: {prediction.replace('_', ' ').title()} ({confidence*100:.2f}% confidence)")

            st.markdown("---")
            st.subheader("GradCAM Heatmap")
            gradcam = generate_gradcam_heatmap(img, model)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.imshow(gradcam, cmap='jet', alpha=0.4)
            ax.axis('off')
            st.pyplot(fig)

            img_base64 = encode_image_base64(img)
            text_prompt = build_plain_text_prompt(prediction, confidence*100)

            # === Consult Claude
            if mode == "Single Model":
                try:
                    answer = ask_claude_bedrock_multimodal(selected_model_id, img_base64, text_prompt)
                    st.markdown("---")
                    st.subheader(f"🧠 AI Explanation ({selected_model_label})")
                    st.markdown(
                        f"""
                        <div style="background-color:#eef6fb;padding:15px;border-radius:10px;">
                        {answer}
                        </div>
                        """, unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"❌ Error querying {selected_model_label}: {e}")

            elif mode == "Side-by-Side Comparison":
                try:
                    answer_3_7 = ask_claude_bedrock_multimodal(model_options["Claude 3.7 Sonnet"], img_base64, text_prompt)
                except Exception as e:
                    answer_3_7 = f"❌ Error querying Claude 3.7: {e}"

                try:
                    answer_3_5 = ask_claude_bedrock_multimodal(model_options["Claude 3.5 Sonnet"], img_base64, text_prompt)
                except Exception as e:
                    answer_3_5 = f"❌ Error querying Claude 3.5: {e}"

                st.markdown("---")
                st.subheader("🧠 Side-by-Side AI Explanations")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Claude 3.7 Sonnet")
                    st.markdown(
                        f"""
                        <div style="background-color:#eef6fb;padding:15px;border-radius:10px;">
                        {answer_3_7}
                        </div>
                        """, unsafe_allow_html=True
                    )

                with col2:
                    st.markdown("### Claude 3.5 Sonnet")
                    st.markdown(
                        f"""
                        <div style="background-color:#eef6fb;padding:15px;border-radius:10px;">
                        {answer_3_5}
                        </div>
                        """, unsafe_allow_html=True
                    )
