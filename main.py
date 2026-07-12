from ultralytics import YOLO
import cv2

# Load YOLOv8 model (auto downloads, no manual files needed 😎)
model = YOLO("yolov8n.pt")   # nano version = fast

# Open video
cap = cv2.VideoCapture("input/56310-479197605.mp4")

# Line position
line_y = 300

# Counter
count = 0
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])

        # COCO classes: car=2, bus=5, truck=7
        if cls in [2, 5, 7]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Count logic
            if cy > line_y and i not in counted_ids:
                count += 1
                counted_ids.add(i)

    # Draw line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    # Show count
    cv2.putText(frame, f"Count: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Vehicle Counter YOLOv8", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()