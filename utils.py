# utils.py
import cv2
import numpy as np

def process_frame(frame, net, output_layers, width, height, line_y, detected_ids, count):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    centers = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id in [2, 5, 7] and confidence > 0.5:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(cx - w/2)
                y = int(cy - h/2)

                boxes.append((x, y, w, h))
                centers.append((cx, cy))

    for i, (cx, cy) in enumerate(centers):
        if cy > line_y and i not in detected_ids:
            count += 1
            detected_ids.add(i)

    return boxes, centers, count