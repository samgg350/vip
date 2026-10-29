"""Build a YOLO dataset for the yellow lime toy, labeled automatically by color.

1. Adjust the sliders until ONLY the lime is white in the Mask window
   (defaults are the yellow values from color_seeker.py).
2. Press SPACE to start/stop auto-saving. Move the lime around: near, far,
   left, right, different backgrounds and lighting. Aim for 150+ images.
3. Press N to save a background frame with NO lime (teaches "nothing here").
   Take the lime out of view first. 20-30 of these is plenty.
4. Press Q when done, then run: python train_lime.py
"""
import os
import random
import time
import cv2
import numpy as np
import requests
import urllib3

urllib3.disable_warnings()

# -- CONFIGURATION ----------------------------------------------------
# Same as STREAM_URL in obj_track_advpid.py (0 = laptop webcam)
STREAM_URL = "https://192.168.1.190:8080/camera/stream"
DATASET_DIR = "dataset"
CLASS_NAME = "lime"
SAVE_INTERVAL = 0.3   # seconds between auto-saved images
MIN_AREA = 300        # ignore blobs smaller than this (noise)
VAL_FRACTION = 0.2    # share of images used for validation
# --------------------------------------------------------------------


def frames(url):
    if isinstance(url, int):
        cap = cv2.VideoCapture(url)
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    resp = requests.get(url, stream=True, timeout=10, verify=False)
    buf = b""
    for chunk in resp.iter_content(4096):
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                buf = buf[-1:]
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                buf = buf[start:]
                break
            jpg = buf[start:end + 2]
            buf = buf[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def find_lime(frame, lower, upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return None, mask
    return cv2.boundingRect(biggest), mask


def save(frame, box):
    split = "val" if random.random() < VAL_FRACTION else "train"
    name = f"{int(time.time() * 1000)}"
    cv2.imwrite(os.path.join(DATASET_DIR, "images", split, name + ".jpg"), frame)
    with open(os.path.join(DATASET_DIR, "labels", split, name + ".txt"), "w") as f:
        if box is not None:
            h, w = frame.shape[:2]
            x, y, bw, bh = box
            # YOLO format: class x_center y_center width height (all 0..1)
            f.write(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")


def count_images():
    return sum(len(os.listdir(os.path.join(DATASET_DIR, "images", s))) for s in ("train", "val"))


def main():
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(DATASET_DIR, kind, split), exist_ok=True)
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(DATASET_DIR)}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"names:\n  0: {CLASS_NAME}\n")

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    for name, val, mx in [("H min", 24, 179), ("H max", 50, 179), ("S min", 96, 255),
                          ("S max", 255, 255), ("V min", 100, 255), ("V max", 255, 255)]:
        cv2.createTrackbar(name, "Controls", val, mx, lambda _: None)

    print(__doc__)
    recording = False
    last_save = 0.0
    saved = count_images()

    for frame in frames(STREAM_URL):
        g = lambda n: cv2.getTrackbarPos(n, "Controls")
        lower = np.array([g("H min"), g("S min"), g("V min")])
        upper = np.array([g("H max"), g("S max"), g("V max")])
        box, mask = find_lime(frame, lower, upper)

        if recording and box is not None and time.time() - last_save > SAVE_INTERVAL:
            save(frame, box)
            last_save = time.time()
            saved += 1

        view = frame.copy()
        if box is not None:
            x, y, bw, bh = box
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        state = "RECORDING" if recording else "paused (SPACE to record)"
        cv2.putText(view, f"{state} | saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 255), 2)
        cv2.imshow("Camera", view)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            recording = not recording
        if key == ord("n"):
            save(frame, None)
            saved += 1
            print("Saved background frame (no lime).")

    cv2.destroyAllWindows()
    print(f"Done. {saved} images in '{DATASET_DIR}'. Next: python train_lime.py")


if __name__ == "__main__":
    main()
