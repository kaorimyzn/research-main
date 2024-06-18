import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model and labels
model = tf.keras.models.load_model('grape.h5')
with open('labels.txt', 'r') as f:
    class_labels = f.read().splitlines()

# Define the Streamlit app
st.title("Deteksi Penyakit Tanaman Anggur")
st.write("Unggah gambar daun anggur untuk mendeteksi penyakit.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Display the top 3 predicted classes
    top_k = 3
    top_classes = np.argsort(predictions[0])[::-1][:top_k]
    st.write("Top Predictions:")
    for i, class_index in enumerate(top_classes):
        class_label = class_labels[class_index]
        confidence = predictions[0][class_index]
        st.write(f"{i+1}. {class_label} ({confidence:.2f})")

