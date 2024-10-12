import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Load the model
model = load_model('./deepfakeImg_detector_1.keras')

# Function to process the uploaded image
def process_image(uploaded_file):
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(128, 128))  # Resize the image to 128x128
    img_array = img_to_array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Rescale the image (normalize)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return img_array

st.title('Deepfake Detection App')

# File uploader for image input
uploaded_file = st.file_uploader('Choose an image among ".jpg", ".jpeg", ".png", ".webp" formats...', type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Button to predict the class
    if st.button('Predict'):
        # Process the image and make predictions
        image = process_image(uploaded_file)
        prediction = model.predict(image)

        # Interpret the prediction
        if prediction[0][0] > 0.5:
            result = "Real"
            probability = prediction[0][0]
        else:
            result = "Fake"
            probability = 1-prediction[0][0]

        # Display the prediction result
        st.write(f'Prediction: {probability*100:.2f}% chances of being "{result}".')
