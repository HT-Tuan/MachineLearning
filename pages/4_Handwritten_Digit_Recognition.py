import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2


st.set_page_config(
    page_title="Nhận dạng chữ số viết tay",
    page_icon="✍️"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #e88102;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title('Nhận dạng chữ số viết tay')

result_image = np.zeros((224, 224, 3), dtype=np.uint8)

image_element = st.image(result_image, channels='RGB')

# Load model
model_architecture = "./src/Handwritten_Digit_Recognition/digit_config.json"
model_weights = "./src/Handwritten_Digit_Recognition/digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test

RESHAPED = 784

X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

X_test /= 255

if 'index' not in st.session_state:
    st.session_state.index = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 's' not in st.session_state:
    st.session_state.s = ''
else:
    st.session_state.s = ''
left_column, right_column = st.columns(2)

with left_column:
    if st.button('Tạo ảnh ngẫu nhiên'):
        st.session_state.index = np.random.randint(0, 9999, 150)
        digit_random = np.zeros((10*28, 15*28), dtype=np.uint8)
        for i in range(0, 150):
            m = i // 15
            n = i % 15
            digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[st.session_state.index[i]] 
        cv2.imwrite('./images/digit_random.jpg', digit_random)

        st.session_state.image = cv2.imread('./images/digit_random.jpg')
        image_element.image(st.session_state.image, channels='RGB')

with right_column:
    if st.button('Nhận dạng'):
        image_element.image(st.session_state.image, channels='RGB')
        X_test_sample = np.zeros((150,784), dtype=np.float32)
        for i in range(0, 150):
            X_test_sample[i] = X_test[st.session_state.index[i]] 

        prediction = model.predict(X_test_sample)
        st.session_state.s = ''
        for i in range(0, 150):
            ket_qua = np.argmax(prediction[i])
            st.session_state.s = st.session_state.s + str(ket_qua) + ' '
            if (i+1) % 15 == 0:
                st.session_state.s = st.session_state.s + '\n'
with left_column:
    st.text(st.session_state.s)