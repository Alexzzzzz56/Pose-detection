import streamlit as st
# HKU SPACE CCIT4080A
# Team: ∀ ASS
# FINISHED by <Li Ho Yin>
st.set_page_config(page_title="Source Code", page_icon="For_all_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_all_ASS.jpeg")
col2.title("Source Code")
st.header("", divider="red")
Pages = st.selectbox("Pages select:",("Home","Pose Detection", "Upload file", "Source Code"))
if Pages == "Home":
    code = '''import streamlit as st
import os
# HKU SPACE CCIT4080
# Team: ∀ ASS
# FINISHED by <Li Ho Yin>


st.set_page_config(page_title="∀ ASS", page_icon="For_all_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_all_ASS.jpeg")
col2.title("∀ ASS")
with st.sidebar:
    st.image("For_all_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1, 1, 1, 1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chan")
    st.header("", divider="red")
st.header("", divider=True)
st.subheader("Hello! We are team _∀ ASS_ and the students from HKU SPACE!")
st.image("Project_idea.jpg", caption="∀ ASS Project idea")'''
    st.code(code)
if Pages == "Pose Detection":
    code = '''import cv2
import numpy as np
import av
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import (webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase, VideoTransformerBase)
import queue
from typing import List, NamedTuple
import pandas as pd
# HKU SPACE CCIT4080
# Team: ∀ ASS
# FINISHED by <Li Ho Yin>

# The Streamlit library is used to build the web UI.
# The page title, icon, and layout are configured.
# The user can select the Movenet model and set the confidence threshold using sliders.
st.set_page_config(page_title="Pose Detection", page_icon="For_all_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_all_ASS.jpeg")
col2.title("Pose Detection")
menu_option = st.selectbox("Movenet Model Select:", (
    "Movenet lightning (float 16)",
    "Movenet thunder (float 16)",
    "Movenet lightning (int 8)",
    "Movenet thunder (int 8)"))
th1 = st.slider("confidence threshold", 0.0, 1.0, 0.3, 0.05)
st.caption("Suggest the confidence threshold should be setted between 0.3 to 0.4 to get the best result")
expander = st.expander("the usage of the confidence threshold")
expander.write("Based on Movenet model, the output (17 key points) will have its own confidence threshold, if the output's confidence threshold is larger than the custom confidence threshold, then that key point and its connection line will be drawn.")
code_th1 = ...
expander.code(code_th1, language='python')

# The load_model function is defined to load the Movenet model based on the user's selection.
# The function returns the input size and the path to the model file.
def load_model(menu_option):
    if menu_option == "Movenet lightning (float 16)":
        interpreter = "lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"
        input_size = 192

    elif menu_option == "Movenet thunder (float 16)":
        interpreter = "lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
        input_size = 256

    elif menu_option == "Movenet lightning (int 8)":
        interpreter ="lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        input_size = 192

    elif menu_option == "Movenet thunder (int 8)":
        interpreter ="lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"
        input_size = 256

    return input_size,interpreter

# The Movenet function is defined to perform pose detection on a single frame.
# It takes the frame as input, resizes it to the required input size, and runs it through the Movenet model.
# The function returns the key points with their corresponding confidence scores.
def Movenet(Frame):
    input_size, option = load_model(menu_option)
    Frame = tf.expand_dims(Frame, axis=0)
    Frame = tf.image.resize_with_pad(Frame, input_size, input_size)
    interpreter = tf.lite.Interpreter(model_path=option)
    interpreter.allocate_tensors()
    input_image = tf.cast(Frame, dtype=tf.uint8)
    input_detail = interpreter.get_input_details()
    output_detail = interpreter.get_output_details()
    interpreter.set_tensor(input_detail[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_detail[0]['index'])
    return keypoints_with_scores

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
KEYPOINT = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()



# The video_frame_callback function is defined to process each video frame captured from the webcam.
# It calls the Movenet function to perform pose detection on the frame and draws the detected key points and connections on the frame.

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    keypoint_with_scores = Movenet(image)
    x, y, c = image.shape
    keypoint_with_scores = np.multiply(keypoint_with_scores, [x, y, 1])
    shaped = np.squeeze(keypoint_with_scores)
    result_queue.put(keypoint_with_scores)

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > th1:
            cv2.circle(image, (int(kx), int(ky)), 3, (255, 255, 255), -1)

    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > th1) & (c2 > th1):
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)


    return av.VideoFrame.from_ndarray(image, format="bgr24")



webRTC =webrtc_streamer(key="Pose Detection",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

if st.checkbox("Show more", value=True):
    if webRTC.state.playing:
        keypoint_message = st.empty()
        while True:
            result = result_queue.get()
            more_result = pd.DataFrame({
                "Keypoints": KEYPOINT,
                "X Coordinate": result[:, 1],
                "Y Coordinate": result[:, 0],
                "confidence threshold": result[:, 2]
            })
            keypoint_message.table(more_result)



st.markdown("This demo uses a model and code from")
st.markdown("https://tfhub.dev/google/movenet/singlepose/lightning/4")
st.markdown("https://tfhub.dev/google/movenet/singlepose/thunder/4")
st.markdown("Many thanks to the project")'''
    st.code(code)
if Pages == "Upload file":
    code = '''import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# HKU SPACE CCIT4080
# Team: ∀ ASS
# FINISHED by <Li Ho Yin>
st.set_page_config(page_title="Upload file", page_icon="For_all_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_all_ASS.jpeg")
col2.title("Upload file")
st.header("", divider="red")
st.error("Still need develop, don't use!!!!!!!!! ")
uploader_files = st.file_uploader("Please upload a photo with single person and whole body ", type= ['png','jpg'])
menu_option = st.selectbox("Movenet Model Select:", (
    "Movenet lightning (float 16)",
    "Movenet thunder (float 16)",
    "Movenet lightning (int 8)",
    "Movenet thunder (int 8)"))
th1 = st.slider("confidence threshold", 0.0, 1.0, 0.3, 0.05)

def load_model(menu_option):
    if menu_option == "Movenet lightning (float 16)":
        interpreter = "lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"
        input_size = 192

    elif menu_option == "Movenet thunder (float 16)":
        interpreter = "lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"
        input_size = 256

    elif menu_option == "Movenet lightning (int 8)":
        interpreter ="lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite"
        input_size = 192

    elif menu_option == "Movenet thunder (int 8)":
        interpreter ="lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"
        input_size = 256

    return input_size,interpreter

def Movenet(Frame):
    input_size, option = load_model(menu_option)
    Frame = tf.expand_dims(Frame, axis=0)
    Frame = tf.image.resize_with_pad(Frame, input_size, input_size)
    interpreter = tf.lite.Interpreter(model_path=option)
    interpreter.allocate_tensors()
    input_image = tf.cast(Frame, dtype=tf.uint8)
    input_detail = interpreter.get_input_details()
    output_detail = interpreter.get_output_details()
    interpreter.set_tensor(input_detail[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_detail[0]['index'])
    return keypoints_with_scores
KEYPOINT = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder",
                "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee",
                "right knee", "left ankle", "right ankle"]
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
results = []
#file_name = st.text_input("your file name")
#st.download_button("all Processed data", results, file_name=f"train_data.csv")
i = 0
if uploader_files is not None:
    for uploader_file in uploader_files:
        st.error("I told you don't use leh, You see, Error!!!!!!")
        i += 1
        uploader_file = tf.io.read_file(uploader_file)
        uploader_file = tf.image.decode_png(uploader_file)
        st.image(uploader_file)'''
    st.code(code)
if Pages == "Source Code":
    st.info("Well, you cannot see me in me, right?")
