import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cifar10_classifier.hdf5')
    return model

model = load_model()

st.write("""
# CIFAR-10 Image Classification System
""")

file = st.file_uploader("Upload Image", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (32, 32)  # CIFAR-10 images are 32x32
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image to match the model's training data
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
