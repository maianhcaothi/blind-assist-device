from ultralytics import YOLO
from gtts import gTTS
import urllib.request
import numpy as np
import cv2
import concurrent.futures
import os
import time

# URL of your IP cam
url = 'http://172.20.10.2/cam-hi.jpg'

# Load YOLOv11 model
model = YOLO("yolo11n.pt")

def fetch_frame():
    try:
        img_resp = urllib.request.urlopen(url, timeout=5)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

def speak_detections(detections):
    if not detections:
        return
    output_text = " ".join(detections)
    tts = gTTS(output_text)
    tts.save("gg.mp3")
    # os.system("start gg.mp3")  # Windows
    # os.system("afplay gg.mp3")  # macOS
    os.system("mpg123 gg.mp3")  # Linux

def detect_and_announce(frame):
    results = model(frame)[0]
    detections = set()
    for box in results.boxes:
        detections.add(model.names[int(box.cls[0])])
    print("Detected:", detections)
    speak_detections(detections)

def main():
    print("Starting detection. Press 'q' to quit.")
    cv2.namedWindow("YOLOv11 Detection", cv2.WINDOW_AUTOSIZE)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            future = executor.submit(fetch_frame)
            frame = future.result()

            if frame is not None:
                # Run detection
                detect_and_announce(frame)
                break
                # Optionally show the image
                cv2.imshow("YOLOv11 Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received. Retrying...")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
