import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# Page config
st.set_page_config(page_title="Parkinson Detection", layout="centered")

# Title
st.title("Parkinson’s Disease Detection")
st.write("Upload a drawing image to check if it indicates Parkinson’s disease.")

# Load model
model = joblib.load("model.pkl")

IMG_SIZE = (128, 128)

# Feature extraction (same as your training)
def extract_features(img):
    img = cv2.resize(img, IMG_SIZE)

    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), feature_vector=True)

    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist/(hist.sum()+1e-8)

    edges = cv2.Canny(img,50,150)
    edge_density = np.sum(edges>0)/img.size

    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    return np.concatenate([hog_feat, hist,
                           [edge_density, mean_intensity, std_intensity]])

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = extract_features(img)
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]

    # Probability (if available)
    try:
        prob = model.predict_proba(features)[0][1]
    except:
        prob = None

    st.subheader("Result")

    if prediction == 0:
        st.success("Healthy")
    else:
        st.error("Parkinson Detected")

    if prob is not None:
        st.write(f"Confidence: {prob:.2f}")