import streamlit as st
import torch
import cv2
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Nhận dạng trái cây",
    page_icon="🍎"
)
css = """
    <style>
        .css-6qob1r {
            background-color: #e88102;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)
# Load YOLOv5 model
model = torch.hub.load('./src/NhanDangTraiCay/yolov5', 'custom', path='./src/NhanDangTraiCay/models/best.pt', source='local', force_reload=True)
model.eval()
model.conf = 0.8

# Function for fruit detection and display
def detect_and_display_fruits(image):
    results = model(image)
    results.print()

    r_img = results.render()
    img_result = r_img[0]
    st.image(img_result, caption='Fruit Detection Result', use_column_width=True)

# Main Streamlit page
def main():
    st.title("Nhận dạng trái cây")

    # Upload image from device
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        file_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform fruit detection and display the result
        detect_and_display_fruits(image)

if __name__ == "__main__":
    main()
