import cv2
import socket
import numpy as np
import requests
import time
import urllib3
from ultralytics import YOLO

# phonesense serves over HTTPS with a self-signed cert; silence the warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -- CONFIGURATION ----------------------------------------------------
# Paste your phone stream URL here
# Android (IP Webcam):       "http://192.168.x.x:8080/video"
# iPhone (Simple IP Camera): "http://192.168.x.x:8080/live"
# phonesense:                "https://192.168.x.x:8080/camera/stream"
# Laptop webcam (testing):   0
STREAM_URL = "https://192.168.1.190:8080/camera/stream"

# Robot. Same values as drive_control.py / color_seeker.py.
ARDUINO_IP = "192.168.1.217"
CMD_PORT = 5005               # udp_command_receiver.ino listens on 5005
ROBOT_ENABLED = False         # False = detection only, no packets sent

# Motor pins on the SnappyXO shield (sent inside each "M" command)
LEFT_IN1, LEFT_IN2, LEFT_PWM = 4, 5, 6       # MOTOR #1
RIGHT_IN1, RIGHT_IN2, RIGHT_PWM = 12, 13, 11  # MOTOR #2
LEFT_DIR = 1                  # set to -1 if the left wheel spins backwards
RIGHT_DIR = 1                 # set to -1 if the right wheel spins backwards

# Path to your trained model from Exercise A (train_lime.py makes best.pt)
MODEL_PATH = "best.pt"
# Only track these class names. None = any class in the model.
# No best.pt yet? Use MODEL_PATH = "yolo26n.pt" and
# TARGET_CLASSES = ["sports ball", "orange", "apple"]
TARGET_CLASSES = None

# Only show detections above this confidence level (0.0 to 1.0)
CONFIDENCE = 0.4
SPEED = 0                     # start at 0 (turn only), then ~200
TURN = 255
MIN_PWM = 190                 # smallest PWM that actually turns your motors
DEADBAND = 0.1                # ignore error smaller than this (target ~centered)
STOP_WHEN_LOST = True         # stop motors when the target is not visible
SEND_INTERVAL = 0.1           # seconds between motor packets
# --------------------------------------------------------------------

cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_last_send = 0.0


class PID:
    """Proportional-Integral-Derivative controller with optional output clamping."""

    def __init__(self, kp, ki, kd, max_integral=100.0, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error):
        now = time.time()
        dt = 0.02 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time = now

        if dt > 2:
            self._integral = 0

        self._integral = np.clip(
            self._integral + error * dt, -self.max_integral, self.max_integral
        )
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)
        return output

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


def _motor(in1, in2, pwm, speed):
    if ROBOT_ENABLED:
        msg = f"M {in1} {in2} {pwm} {int(speed)}"
        cmd_sock.sendto(msg.encode(), (ARDUINO_IP, CMD_PORT))


def _boost(speed):
    """Push small non-zero speeds up to MIN_PWM so the motor actually turns."""
    if speed == 0:
        return 0
    sign = 1 if speed > 0 else -1
    return sign * min(255, max(abs(speed), MIN_PWM))


def send_command(left, right):
    global _last_send
    now = time.time()
    if now - _last_send < SEND_INTERVAL:
        return
    _last_send = now
    _motor(LEFT_IN1, LEFT_IN2, LEFT_PWM, LEFT_DIR * _boost(left))
    _motor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, RIGHT_DIR * _boost(right))


def send_stop():
    # never throttled, so a stop is never skipped
    _motor(LEFT_IN1, LEFT_IN2, LEFT_PWM, 0)
    _motor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, 0)


def open_stream(url):
    print(f"Connecting to stream: {url}")
    if isinstance(url, int):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ConnectionError(f"Could not open webcam {url}")
        print("Webcam opened.")
        return cap
    response = requests.get(url, stream=True, timeout=10, verify=False)
    if response.status_code != 200:
        raise ConnectionError(f"Could not connect. Status: {response.status_code}")
    print("Stream connected.")
    return response


def read_frames(stream_response):
    """Yield decoded JPEG frames from an MJPEG stream (or webcam frames).

    Iterates the response exactly once and keeps a persistent buffer so JPEG
    boundaries that span chunks (or frames) are never dropped. Re-calling
    ``iter_content`` per frame would raise ``StreamConsumedError``.
    """
    if isinstance(stream_response, cv2.VideoCapture):
        while True:
            ok, frame = stream_response.read()
            if not ok:
                return
            yield frame

    buf = b""
    for chunk in stream_response.iter_content(4096):
        if not chunk:
            continue

        buf += chunk

        # A single chunk can hold several complete JPEGs; drain them all.
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                # No frame start yet. Keep one byte in case the marker is
                # split across chunks.
                buf = buf[-1:]
                break

            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                # Partial JPEG -- keep it and wait for more bytes.
                buf = buf[start:]
                break

            jpg = buf[start : end + 2]
            buf = buf[end + 2 :]

            if len(jpg) <= 2:
                continue

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def findbest(results, names):
    """Return the bounding box of the largest detected target, or None."""
    best = None
    best_area = 0

    for r in results:
        for box in r.boxes:
            if TARGET_CLASSES and names[int(box.cls[0])] not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)

    return best


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded. Classes:", model.names)
    print("Robot:", f"{ARDUINO_IP}:{CMD_PORT}" if ROBOT_ENABLED else "DISABLED (detection only)")

    stream = open_stream(STREAM_URL)

    print("Press Q to quit.")
    fps_timer = time.time()
    frame_count = 0
    fps = 0

    pid = PID(0.7, 0.1, 0.05, output_limit=1)

    try:
        for frame in read_frames(stream):
            results = model(frame, imgsz=416, conf=CONFIDENCE, verbose=False)
            annotated = results[0].plot()
            wid = len(frame[0])

            best = findbest(results, model.names)
            if best is not None:
                x = (best[0] + best[2]) / 2
                x = (x - wid / 2) / (wid / 2)
                if abs(x) < DEADBAND:
                    x = 0
                power = pid.update(x) * TURN
                left, right = SPEED + power, SPEED - power
                send_command(left, right)
                cv2.circle(annotated, (int((best[0] + best[2]) / 2), int((best[1] + best[3]) / 2)),
                           6, (0, 0, 255), -1)
                status = f"TRACK err={x:+.2f} L={int(left)} R={int(right)}"
            else:
                pid.reset()
                if STOP_WHEN_LOST:
                    send_stop()
                status = "NO TARGET"

            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()
            cv2.line(annotated, (wid // 2, 0), (wid // 2, len(frame)), (255, 255, 255), 1)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Phone Stream - YOLO Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        send_stop()
        if isinstance(stream, cv2.VideoCapture):
            stream.release()
        else:
            stream.close()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
