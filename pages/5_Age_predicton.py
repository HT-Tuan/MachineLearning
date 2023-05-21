import streamlit as st
import numpy as np
import cv2 as cv
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os


st.set_page_config(
    page_title="Dự đoán tuổi",
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

url = 'https://drive.google.com/uc?id=18E4OPrtv2OT6pbFX-vIToda1QqGm942u'
output = './src/Age_Prediction/models/age_model_weights.h5'
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


st.title('Dự đoán tuổi')
FRAME_WINDOW = st.image([])
isCamera = False

deviceId = 0
cap = cv.VideoCapture(deviceId)

if(cap.isOpened()):
    isCamera = True
else:
    isCamera = False

if isCamera == True and 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

if isCamera == True and st.button('Stop'):
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

    print('Trang thai nhan Stop', st.session_state.stop)


if isCamera == True and 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('./images/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')


def visualize(input, faces, fps, age_model, output_indexes, thickness=2):
    dem = 0
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x = coords[0]
            y = coords[1]
            w = coords[2]
            h = coords[3]
            cv.rectangle(input, (x, y), (x + w, y + h), (0, 255, 0), thickness)
            detected_face = input[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            try:
                margin = 30
                margin_x = int((w * margin) / 100)
                margin_y = int((h * margin) / 100)
                detected_face = input[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
            except:
                print("detected face has no margin")

            try:
                detected_face = cv.resize(detected_face, (224, 224))
                img_pixels = img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                age_distributions = age_model.predict(img_pixels)
                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))
                cv.putText(input, apparent_age, (x + int(w / 2), y - 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
            except Exception as e:
                print("Exception:", str(e))
            dem = dem + 1

    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Load the age model
def ageModel():
    json_file = open('./src/Age_Prediction/models/age_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    age_model = model_from_json(loaded_model_json)

    age_model.load_weights("./src/Age_Prediction/models/age_model_weights.h5")

    return age_model


age_model = ageModel()
output_indexes = np.array([i for i in range(0, 101)])

# Load the face detection model
detector = cv.FaceDetectorYN.create(
    './src/Face_detection/face_detection_yunet_2022mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

tm = cv.TickMeter()

if isCamera == False:
    camera_st = st.camera_input(label="CAMERA")

    if camera_st is not None:
        bytes_data = camera_st.getvalue()
        img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
        height, width, channels = img.shape

        frameWidth = int(width)
        frameHeight = int(height)
        detector.setInputSize([frameWidth, frameHeight])

        frame = cv.resize(img, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        # Draw results on the input image including age prediction
        visualize(frame, faces, tm.getFPS(), age_model, output_indexes)

        # Display the frame
        FRAME_WINDOW.image(frame, channels='BGR')

else:
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])
    while True:
        if st.session_state.stop:
            break

        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        # Draw results on the input image including age prediction
        visualize(frame, faces, tm.getFPS(), age_model, output_indexes)

        # Display the frame
        FRAME_WINDOW.image(frame, channels='BGR')
