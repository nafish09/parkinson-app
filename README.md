# Parkinson’s Disease Detection Using Imaging

## Overview

Detects Parkinson’s disease from drawing images using machine learning and deep learning models. Includes a web app for real-time prediction.

## Features

* EDA: class distribution, sample images, edge density
* Feature engineering: HOG, LBP, edge features
* Models: SVM, Random Forest, XGBoost, KNN, CNN (comparison)
* 5-fold stratified cross-validation
* Final evaluation on unseen test set
* Streamlit web app

## Results

* SVM: ~0.75 accuracy (best)
* KNN: ~0.73
* XGBoost: ~0.70
* CNN: ~0.55–0.60

## Tech Stack

Python, NumPy, OpenCV, scikit-learn, TensorFlow/Keras, Streamlit

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

* app.py
* model.pkl, scaler.pkl
* notebook.ipynb
* requirements.txt

Live App: https://your-streamlit-link
