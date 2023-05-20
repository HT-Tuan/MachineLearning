import streamlit as st
import numpy as np
import cv2
from keras.models import model_from_json

st.set_page_config(
    page_title="Nhận diện cảm xúc",
    page_icon="😃"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #98EECC;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title('Nhận diện cảm xúc')
emotion_dict = {0: "TucGian", 1: "ChanNan", 2: "SoHai", 3: "HanhPhuc", 4: "BinhThuong", 5: "Buon", 6: "NgacNhien"}

json_file = open('./src/NhanDangCamXuc/models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("./src/NhanDangCamXuc/models/emotion_model_weights.h5")

detector = cv2.FaceDetectorYN.create(
    './src/NhanDangCamXuc/models/face_detection_yunet_2022mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000)

cap = cv2.VideoCapture(0)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])

FRAME_WINDOW = st.image([])

stop_button = st.button("Stop")

isCamera = cap.isOpened()

while True:
    if stop_button:
        break

    if isCamera:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        frame = cv2.resize(frame, (frameWidth, frameHeight))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(frame)
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x = coords[0]
                y = coords[1]
                w = coords[2]
                h = coords[3]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame, channels='BGR')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        camera_st = st.camera_input(label="CAMERA")
        if camera_st is not None:
            frame = cv2.imdecode(np.frombuffer(camera_st.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (frameWidth, frameHeight))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect(frame)
            if faces[1] is not None:
                for idx, face in enumerate(faces[1]):
                    coords = face[:-1].astype(np.int32)
                    x = coords[0]
                    y = coords[1]
                    w = coords[2]
                    h = coords[3]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(frame, channels='BGR')
