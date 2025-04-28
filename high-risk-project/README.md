# 🫁 Lung Histopathology Cancer Detection and Explanation using Deep Learning and LLMs

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazonaws)
![Status](https://img.shields.io/badge/Project-High--Risk--Prototype-yellow)

> An interpretable AI system combining CNN classification, GradCAM visualizations, and Claude LLM-based clinical reasoning for lung cancer detection from histopathology images.

---

## 🚀 Project Structure


## 🚀 Project Structure

| File/Folder | Description |
|:------------|:------------|
| `.streamlit/` | Configuration folder for Streamlit secrets |
| `data/data_raw/` | Raw images organized into `lung_aca`, `lung_n`, `lung_scc` classes |
| `data/train/`, `data/val/` | Split training and validation datasets |
| `app.py` | Streamlit app for upload, prediction, GradCAM and LLM commentary |
| `prepare-dataset.py` | Script to split raw data into training and validation |
| `lung_cancer_classifier.py` | Training script for ResNet-18 model |
| `best_lung_cancer_classifier.pt` | Best performing model checkpoint |
| `final_lung_cancer_classifier.pt` | Final model checkpoint |
| `training_metrics.png` | Plot showing accuracy and loss during training |
| `requirements.txt` | List of required Python packages |
| `README.md` | Project overview and setup instructions |

---

## 🖼️ How the System Works

1. **Image Upload**: Upload a histopathology image via the Streamlit app.
2. **CNN Prediction**: Classify the image into:
   - Lung Adenocarcinoma
   - Lung Squamous Cell Carcinoma
   - Normal Lung Tissue
3. **GradCAM Heatmap**: Visualize model attention regions influencing the prediction.
4. **LLM Explanation**: Generate human-readable clinical commentary using Claude 3.7 or 3.5 through AWS Bedrock.

---

## 📊 Model Training

- Model: **ResNet-18** (pretrained and fine-tuned)
- Optimizer: **Adam** (`lr=1e-4`)
- Loss: **Cross Entropy Loss**
- Epochs: **10**
- Device: **MPS (Apple Silicon)** or CPU fallback
- Achieved **~99.9% validation accuracy**
- Training metrics (accuracy/loss) plotted and saved.

---

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/amit94ece/lung-histopathology-ai.git
   cd lung-histopathology-ai
Install dependencies:

bash
```
pip install -r requirements.txt
```

## Prepare the dataset:

Organize raw images into a data_raw/ folder by class.

Run the dataset preparation script:

bash
```
python prepare-dataset.py
```

## Train the CNN model:

bash
```
python lung_cancer_classifier.py
```

## Start the Streamlit app:

bash
```
streamlit run app.py
```

## 🔐 Important Notes
AWS Bedrock credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) must be configured in .streamlit/secrets.toml.

### Claude models used:

Claude 3.7 Sonnet

Claude 3.5 Sonnet

Bedrock region used: us-east-1

## 📚 Requirements
Python 3.11+

PyTorch >= 2.0.0

Torchvision

Streamlit

boto3

numpy

matplotlib

Pillow

tqdm

fpdf

(Full list available in requirements.txt)

## 📷 Screenshots

CNN Prediction and GradCAM Visualization	

LLM-Generated Clinical Commentary
![alt text](image.png)

## 📌 Acknowledgements
Dataset: Lung and Colon Histopathological Image Dataset

Model Backbone: TorchVision ResNet-18

LLMs: Claude 3.7 Sonnet and Claude 3.5 Sonnet via AWS Bedrock

## ✨ Future Work
Incorporate additional cancer types and tissue modalities.

Integrate GPT-4o or specialized medical LLMs.

Conduct real-world evaluation with pathologists.

Enable clinician feedback loops into the system.