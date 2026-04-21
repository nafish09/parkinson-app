import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Parkinson Detection", page_icon="🧠")

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = (128, 128)

# ---------------- FEATURE FUNCTION ----------------
def extract_features(img):
    img = cv2.resize(img, IMG_SIZE)

    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), feature_vector=True)

    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist/(hist.sum()+1e-8)

    edges = cv2.Canny(img,50,150)
    edge_density = np.sum(edges>0)/img.size

    # must match training
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    return np.concatenate([
        hog_feat,
        hist,
        [edge_density, mean_intensity, std_intensity]
    ])

# ---------------- STYLE ----------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e1e1e;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    margin-top: 10px;
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 class='center'>🧠 Parkinson’s Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>Upload spiral, wave or drawing image</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# ---------------- MAIN ----------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Invalid image file")
    else:
        # -------- PREVIEW --------
        display_img = cv2.resize(img, (300, 300))

        st.markdown("<div class='card center'>", unsafe_allow_html=True)
        st.image(display_img, caption="Preview", width=250)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------- FEATURES --------
        features = extract_features(img)
        features = scaler.transform([features])

        # -------- PREDICTION --------
        with st.spinner("Analyzing image..."):
            prediction = model.predict(features)[0]

            try:
                prob = model.predict_proba(features)[0][1]
            except:
                prob = None

        st.markdown("---")

        # -------- RESULT CARD --------
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if prediction == 0:
            st.markdown("<h3 style='color:green;'>✅ Healthy</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>⚠️ Possible Parkinson’s Pattern Detected</h3>", unsafe_allow_html=True)

        # -------- CONFIDENCE --------
        if prob is not None:
            confidence_percent = prob * 100

            st.write("Confidence Level")
            st.progress(float(prob))

            if confidence_percent > 75:
                st.success(f"{confidence_percent:.1f}% (High Confidence)")
            elif confidence_percent > 55:
                st.warning(f"{confidence_percent:.1f}% (Moderate Confidence)")
            else:
                st.error(f"{confidence_percent:.1f}% (Low Confidence)")

            # -------- INTERPRETATION --------
            if prediction == 1:
                st.write("Model detected irregular stroke patterns typical of Parkinson’s drawings.")
            else:
                st.write("Drawing appears smooth and consistent with healthy patterns.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
