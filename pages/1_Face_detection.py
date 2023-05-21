import streamlit as st
import numpy as np
import cv2 as cv

st.set_page_config(
    page_title="Phát hiện khuôn mặt",
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

st.title('Phát hiện khuôn mặt')
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

if isCamera == True and st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


def visualize(input, faces, fps, thickness=2):
    dem = 0
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
            dem = dem + 1
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv.putText(input, 'Total: {:d}'.format(dem), (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

    if camera_st is not None :
        bytes_data = camera_st.getvalue()
        img = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
        height, width, channels = img.shape

        frameWidth = int(width)
        frameHeight = int(height)
        detector.setInputSize([frameWidth, frameHeight])

        frame = cv.resize(img, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        st.image(frame, channels='BGR')
else:
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
