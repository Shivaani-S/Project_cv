import streamlit as st
import cv2
import pandas as pd
from utils import process_frame

st.set_page_config(page_title="Vehicle Counter", layout="wide")

st.title("🚗 Smart Vehicle Counter Dashboard")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file:
    tfile = open("input/temp.mp4", "wb")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture("input/temp.mp4")

    # Load YOLO
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    count = 0
    line_y = 300
    detected_ids = set()

    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        boxes, centers, count = process_frame(
            frame, net, output_layers, width, height, line_y, detected_ids, count
        )

        for (x,y,w,h) in boxes:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.line(frame, (0, line_y), (width, line_y), (0,255,0), 2)

        cv2.putText(frame, f"Count: {count}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        frame_placeholder.image(frame, channels="BGR")

        data.append(count)
        df = pd.DataFrame(data, columns=["Vehicles"])

        chart_placeholder.line_chart(df)

    cap.release()