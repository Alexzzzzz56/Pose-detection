import streamlit as st
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import base64
from playsound import playsound
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
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )



results = []
#file_name = st.text_input("your file name")
#st.download_button("all Processed data", results, file_name=f"train_data.csv")
i = 0
if uploader_files is not None:
    for uploader_file in uploader_files:
        st.error("HAIYAA!I told you don't use leh, You see, Errrrrrrrrrror!!!!!!You Failure!!!!!!")
        autoplay_audio("Error.mp3")
        i += 1
        uploader_file = tf.io.read_file(uploader_file)
        uploader_file = tf.image.decode_png(uploader_file)
        st.image(uploader_file)

        '''outputs = Movenet(uploader_file)
        x, y, c = uploader_file.shape
        outputs = np.multiply(outputs, [x, y, 1])
        result = pd.DataFrame({
            "Keypoints": KEYPOINT,
            "X Coordinate": outputs[:, 1],
            "Y Coordinate": outputs[:, 0],
            "confidence threshold": outputs[:, 2]
        })
        results.append(result)

        shaped = np.squeeze(outputs)
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > th1:
                cv2.circle(uploader_file, (int(kx), int(ky)), 3, (255, 255, 255), -1)

        for edge, color in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > th1) & (c2 > th1):
                cv2.line(uploader_file, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        container = st.container(True)
        container.image(uploader_file)
        container.table(result)
        container.download_button("Processed Photo", uploader_file)
        container.download_button("Processed data", result)'''