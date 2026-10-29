import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO

# -- CONFIGURATION ----------------------------------------------------
# Paste your phone stream URL here
# Android (IP Webcam):      "http://192.168.x.x:8080/video"
# iOS (Simple IP Camera): "http://192.168.x.x:8080/live"
# phonesense:             "https://192.168.x.x:8080/camera/stream"
STREAM_URL = "https://192.168.1.190:8080/camera/stream"

# Path to your trained model from Exercise A
MODEL_PATH = "best.pt"

# Only show detections above this confidence level (0.0 to 1.0)
CONFIDENCE = 0.5
# --------------------------------------------------------------------

def open_stream(url):
    print(f"Connecting to stream: {url}")
    response = requests.get(url, stream=True, timeout=10)
    if response.status_code != 200:
        raise ConnectionError(f"Could not connect. Status: {response.status_code}")
    print("Stream connected.")
    return response

def read_frame(stream_response):
    buf = b""
    for chunk in stream_response.iter_content(4096):
        if not chunk:
            continue

        buf += chunk
        start = buf.find(b"\xff\xd8")
        if start == -1:
            # Keep one byte in case the JPEG marker is split across chunks.
            buf = buf[-1:]
            continue

        end = buf.find(b"\xff\xd9", start + 2)
        if end == -1:
            # Keep the partial JPEG and wait for more bytes.
            buf = buf[start:]
            continue

        jpg = buf[start:end + 2]
        buf = buf[end + 2:]

        if len(jpg) <= 2:
            continue

        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    return None

def main():
    # Load the trained model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    # Connect to the phone stream
    stream = open_stream(STREAM_URL)

    print("Press Q to quit.")
    fps_timer = time.time()
    frame_count = 0

    while True:
        frame = read_frame(stream)
        if frame is None:
            continue

        # Run detection on the frame
        results = model(frame, imgsz=416, conf=CONFIDENCE, verbose=False)

        # Draw boxes on the frame
        annotated = results[0].plot()

        # Calculate and display FPS
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Phone Stream - YOLO26n Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.close()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    main()