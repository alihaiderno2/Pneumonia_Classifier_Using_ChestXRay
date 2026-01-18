import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_model.h5')
    return model

model = load_model()
st.title("AI pneumonia Doctor")
st.write("Upload a chest X-Ray to detect pneumonia.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is not None:
    image_data = Image.open(file).convert('RGB')
    st.image(image_data,caption="Uploaded X-Ray",use_column_width=True)
    img = image_data.resize((64,64))
    x= image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = x/255.0

    prediction = model.predict(x)
    st.write(f"Confidence Score: {prediction[0][0]:.2f}")
    if prediction[0][0] >0.5:
        st.error(" PNEUMONIA DETECTED")
    else:
        st.success(" NORMAL (Healthy)")