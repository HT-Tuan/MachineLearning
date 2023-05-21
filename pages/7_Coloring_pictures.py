import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Set environment variable to avoid OpenMP runtime warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st.set_page_config(
    page_title="Tô màu cho ảnh trắng đen",
    page_icon="👁️"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #e88102;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Function to perform image colorization
def perform_colorization(image):
    SIZE = 160
    gray_img = []

    image = cv2.resize(image, (SIZE, SIZE))
    image = image.astype('float32') / 255.0
    gray_img.append(img_to_array(image))

    json_file = open('./src/ToMauAnh/models/image_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    images_model = model_from_json(loaded_model_json)

    images_model.load_weights("./src/ToMauAnh/models/images_model_weights.h5")
    print("Loaded model from disk")

    predicted = np.clip(images_model.predict(gray_img[0].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
    return predicted


# Streamlit page
def main():
    st.title("Tô màu ảnh")

    # Upload image from device
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform colorization and display the results
        predicted_image = perform_colorization(image)
        st.image(predicted_image, caption='Predicted Image', use_column_width=True)


if __name__ == "__main__":
    main()
