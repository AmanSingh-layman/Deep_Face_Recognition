import streamlit as st
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from PIL import Image


# Define custom L1 Distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


# Load trained Siamese model with custom layer
MODEL_PATH = "siamesemodel.h5"
siamese_model = load_model(MODEL_PATH, custom_objects={'L1Dist': L1Dist}, compile=False)


# Define verification function
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100)) / 255.0
    return np.expand_dims(image, axis=0)


def verify(model, detection_threshold=0.9, verification_threshold=0.3):
    results = []
    input_img = preprocess("application_data/input_image/input_image.jpg")

    for image in os.listdir("application_data/verification_images"):
        validation_img = preprocess(os.path.join("application_data/verification_images", image))
        result = model.predict([input_img, validation_img])
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold

    return verified


# Streamlit UI
st.set_page_config(page_title="Facial Verification App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Verify", "User Guide"])

if page == "Home":
    st.title("Welcome to the Facial Verification System")
    st.write("This app allows you to verify faces using a trained Siamese model.")
    st.image("images_home_page.jfif", caption="Facial Verification", use_container_width=True)
    st.write("Navigate to 'Upload & Verify' to test the model.")

elif page == "Upload & Verify":
    st.title("Facial Verification System")
    st.write("Upload an image for verification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True, width=50)

        # Save uploaded image
        input_image_path = "application_data/input_image/input_image.jpg"
        image.save(input_image_path)

        # Perform verification
        with st.spinner('Verifying...'):
            verified = verify(siamese_model)
        verification_status = "Verified" if verified else "Unverified"
        st.write(f"Verification Status: {verification_status}")

elif page == "User Guide":
    st.title("User Guide")
    st.write("### How to Use the App")
    st.write("1. Navigate to the 'Upload & Verify' page from the sidebar.")
    st.write("2. Upload an image for verification.")
    st.write("3. The model will compare it with stored verification images.(i.e, of AMAN SINGH)")
    st.write(".......   It always return unverified until image of AMAN SINGH is uploaded.")
    st.write("4. You will see the verification result displayed on the screen and sidebar.")
    st.write("5. Return to 'Home' for more information.")
