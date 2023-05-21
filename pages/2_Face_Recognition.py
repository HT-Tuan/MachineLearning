import streamlit as st
import numpy as np
import cv2 as cv
import joblib


st.set_page_config(
    page_title="Nhận dạng khuôn mặt",
    page_icon="😎"
)

css = """
    <style>
        .css-6qob1r {
            background-color: #e88102;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title('Nhận dạng khuôn mặt')

isCamera = False
FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

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


svc = joblib.load('./src/Face_Recognition/svc.pkl')
mydict = ['BanKiet', 'BanNghia', 'BanThanh', 'Dan', 'DangKhoa',
        'HongSon', 'SangSang', 'ThanhTuan', 'ThayDuc', 'ThoTy']


def visualize(input, faces, fps, thickness=2):
    total = 0
    total_detec = 0
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)

            face_align = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            # ==============================
            img_predict = cv.imread('./src/Face_Recognition/' + result + '.bmp')
            detector.setInputSize([img_predict.shape[1], img_predict.shape[0]])
            face_predict = detector.detect(img_predict)
            if face_predict[1] is not None:
                predict_align = recognizer.alignCrop(img_predict, face_predict[1][0])
                predict_feature = recognizer.feature(predict_align)
                # 
                cosine_similarity_threshold = 0.363
                #
                cosine_score = recognizer.match(face_feature, predict_feature, cv.FaceRecognizerSF_FR_COSINE)
                
                if cosine_score >= cosine_similarity_threshold:
                    total_detec = total_detec + 1
                    cv.putText(input, result, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)           
            # =================================  
            cv.rectangle(input, (coords[0], coords[1]), (coords[0] +
                         coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]),
                      2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]),
                      2, (0, 255, 255), thickness)
            total = total + 1

    cv.putText(input, 'FPS: ', (1, 16),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv.putText(input, '{:.2f}'.format(fps), (50, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv.putText(input, 'Total Detection:', (1, 36), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv.putText(input, '{:d}'.format(total), (130, 36), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv.putText(input, 'Total Recognition:', (1, 56), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv.putText(input, '{:d}'.format(total_detec), (145, 56), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        './src/Face_detection/face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)

    recognizer = cv.FaceRecognizerSF.create(
        './src/Face_Recognition/face_recognition_sface_2021dec.onnx', "")

    tm = cv.TickMeter()
    if(isCamera == False):
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
            faces = detector.detect(frame)  # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            st.image(frame, channels='BGR')
    else:
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            detector.setInputSize([frameWidth, frameHeight])
            # Inference
            tm.start()
            faces = detector.detect(frame)  # faces is a tuple
            tm.stop()
            
            # Draw results on the input image
            visualize(frame, faces, tm.getFPS())

            # Visualize results
            FRAME_WINDOW.image(frame, channels='BGR')

