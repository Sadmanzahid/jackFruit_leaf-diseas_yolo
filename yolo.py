# streamlit_app_modern.py
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ LOAD ONNX MODEL
# ===============================
ONNX_MODEL_PATH = r"best.onnx"
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# ===============================
# 2️⃣ DEFINE PREPROCESSING (CLAHE)
# ===============================
def preprocess_image(image: np.ndarray, size=(224, 224)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    img_resized = cv2.resize(img_clahe, size)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2,0,1))
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

# ===============================
# 3️⃣ CLASS NAMES
# ===============================
CLASS_NAMES = ['burn', 'healthy', 'red_rust', 'spot']

# ===============================
# 4️⃣ STREAMLIT UI
# ===============================
st.set_page_config(page_title="Jackfruit Leaf Disease Classifier", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🍈 Jackfruit Leaf Disease Classifier</h1>", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Upload a leaf image (.jpg, .jpeg, .png).  
    2. Wait for the model to predict the disease.  
    3. Check the confidence bar chart and probabilities.  
    """
)

uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Layout with two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_np, use_column_width=True)
    
    # Preprocess & Predict
    input_tensor = preprocess_image(image_np)
    outputs = session.run(None, {input_name: input_tensor})
    probs = outputs[0][0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    
    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"**Class:** {pred_class}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        
        # Probability bar chart
        st.subheader("Class Probabilities")
        fig, ax = plt.subplots()
        ax.barh(CLASS_NAMES, probs*100, color='forestgreen')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Class")
        for i, v in enumerate(probs*100):
            ax.text(v + 1, i, f"{v:.1f}%", color='black', va='center')
        st.pyplot(fig)
        
        # Expandable detailed probabilities
        with st.expander("Show detailed probabilities"):
            for i, cls in enumerate(CLASS_NAMES):
                st.write(f"{cls}: {probs[i]*100:.2f}%")
    
    st.markdown("---")
    st.info("✅ CLAHE preprocessing applied automatically for better accuracy.")
