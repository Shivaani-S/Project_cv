# Smart Vehicle Detection and Counting System 🚗

This is a mini project I built to detect and count vehicles (cars, buses, trucks) from traffic videos using YOLO. Basically it watches a video, draws boxes around vehicles, and counts them as they cross a line on screen.

I made two versions of it:
- `main.py` – a simple script version using YOLOv8, opens in a normal OpenCV window
- `dashboard.py` – a web dashboard version using YOLOv3 + Streamlit, where you can upload a video from the browser and see live counting with a graph

## Why I made this

Manually counting vehicles from CCTV/traffic footage is boring and takes forever, and it's easy to miscount. So I wanted to try automating it using object detection instead of doing it by hand.

## What it does

- Detects cars, buses and trucks in a video frame by frame
- Draws bounding boxes around each detected vehicle
- Has a counting line — once a vehicle crosses it, it gets counted (only once, not multiple times)
- Shows the live count on screen
- Dashboard version also shows a live chart of the count as the video plays

## Tech I used

- Python
- YOLOv8 (Ultralytics) – for main.py
- YOLOv3 (OpenCV DNN) – for dashboard.py
- OpenCV
- Streamlit
- Pandas, NumPy

## Folder structure
├── input/            → put your input video here
├── output/            → processed output goes here
├── yolo/              → yolov3 weights/cfg files for dashboard.py
├── yolov8n.pt          → yolov8 model (auto downloads too)
├── main.py             → script version
├── dashboard.py         → streamlit dashboard version
├── utils.py              → helper function for dashboard
├── requirements.txt
└── README.md

## How to run it

1. Clone this repo
git clone https://github.com/Shivaani-S/Project_cv.git
cd Project_cv

2. Install the requirements
pip install -r requirements.txt

3. Put your traffic video inside the `input/` folder

4. To run the script version:
python main.py
(press Esc to close the window)

5. To run the dashboard version:
streamlit run dashboard.py
it'll open in your browser, just upload a video and it'll start counting.

Note: for dashboard.py you need yolov3.weights and yolov3.cfg inside the yolo/ folder, it doesn't auto download like yolov8 does.

## How the counting works

Every frame it detects vehicles, finds the center point of each box, and checks if that center has crossed the line (`line_y`). If it has and it wasn't already counted, count goes up by 1. That's it, pretty simple logic.

## Things I want to improve later

- Use a proper tracker (like DeepSORT) instead of the current method, so counting is more accurate when vehicles overlap
- Make both files use the same model instead of two different ones
- Count cars/buses/trucks separately instead of one total number
- Save the results to a CSV file
- Try it on a live camera feed instead of just uploaded videos

## Credits

- Ultralytics YOLOv8 docs
- OpenCV docs
- Streamlit docs
- COCO dataset
