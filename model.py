import streamlit as st
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import numpy as np
import io
import os

from PIL import Image
from fracture_descriptions import fracture_dict
from tensorflow import keras

# Update deprecated numpy types
np.int = np.int64
np.bool = np.bool_

# Check if the model file exists
model_path = 'model/basic_model.h5'

if not os.path.exists(model_path):
    print("Model file not found! Please check the path and ensure the model file is available.")
else:
    print("Model file found, loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

# Dictionary to map class indices to human-readable class names
class_names_dict = {
    0: 'Avulsion fracture',
    1: 'Comminuted fracture',
    2: 'Fracture Dislocation',
    3: 'Greenstick fracture',
    4: 'Hairline Fracture',
    5: 'Impacted fracture',
    6: 'Longitudinal fracture',
    7: 'Oblique fracture',
    8: 'Pathological fracture',
    9: 'Spiral Fracture',
    10: 'non_fractured'
}

def preprocess_single_image(img, target_size=(256, 256)):
    """Load and preprocess a single image."""
    img = img.resize(target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_single_image(model, img_array, class_names_dict):
    """Make a prediction on a single image."""
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=-1)
    class_name = class_names_dict[predicted_class[0]]
    return class_name

def shapley(img_array, col2):
    """Function to display Shapley values."""
    st.write('Loading Shapley... ')
    masker = shap.maskers.Image("blur(16, 16)", img_array[0].shape)
    explainer = shap.Explainer(model, masker, output_names=list(class_names_dict.values()))
    shap_values = explainer(img_array, max_evals=8000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

    # Plot SHAP values
    plt.figure()
    shap.image_plot(shap_values, pixel_values=img_array, show=False)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    col2.image(buf, caption='Shapley Values', use_column_width=True, clamp=True)
    buf.close()

def analyze_fracture(uploaded_file):
    """Analyze the uploaded X-ray image for fractures."""
    col1, col2 = st.columns(2)

    # Turn file into image and display it
    img = Image.open(uploaded_file)
    col1.image(img, caption='Uploaded X-Ray', use_column_width=True, output_format='JPEG', clamp=True)

    img_array = preprocess_single_image(img)
    print("Image preprocessed successfully.")

    predicted_fracture = predict_single_image(model, img_array, class_names_dict)
    print(f"Predicted class: {predicted_fracture}")

    if predicted_fracture == 'non_fractured':
        col2.markdown(f'<div class="text-box">#### No Fracture Detected \n</div>', unsafe_allow_html=True)
    else:
        # Display fracture with description
        fracture_description = fracture_dict[predicted_fracture]  # Get description from fracture_descriptions.py file
        col2.markdown(f'<div class="text-box">#### {predicted_fracture} \n</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="text-box">\n Fracture Description: \n {fracture_description}</div>', unsafe_allow_html=True)

        shapley(img_array, col2)  # Pass col2 to shapley function to display the plot
