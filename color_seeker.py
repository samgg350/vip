import socket
import time
import threading
import cv2
import numpy as np
import requests
import urllib3

urllib3.disable_warnings()   # phonesense uses a self-signed HTTPS certificate

# ===================== CONFIGURATION =====================
ARDUINO_IP = "192.168.1.217"   # same IP you use in drive_control.py
CMD_PORT   = 5005              # your udp_command_receiver.ino uses 5005

# Camera source. Pick ONE:
#   PHONE_URL = "https://192.168.1.190:8080/camera/stream"   # phonesense
#   PHONE_URL = "http://192.168.x.x:8080/video"              # Android IP Webcam
#   PHONE_URL = "http://192.168.x.x:8080/live"               # iPhone Simple IP Camera
#   PHONE_URL = 0                                            # laptop webcam (testing)
PHONE_URL = "https://192.168.1.190:8080/camera/stream"

# Motor pins on the SnappyXO shield
LEFT_IN1,  LEFT_IN2,  LEFT_PWM  = 4, 5, 6      # MOTOR #1
RIGHT_IN1, RIGHT_IN2, RIGHT_PWM = 12, 13, 11   # MOTOR #2

# Motion speeds (0-255)
SCAN_SPEED      = 190   # rotation speed while searching
FORWARD_SPEED   = 210   # base forward speed while pursuing
TURN_GAIN       = 160   # steering strength
FORWARD_FALLOFF = 0.5   # slow down forward motion when object is off-center

# Smallest PWM that actually makes your motors turn (below this they only hum)
MIN_PWM = 190

# Behavior thresholds
MIN_AREA   = 300     # ignore blobs smaller than this (noise)
STOP_AREA  = 7000    # "close enough" when blob area exceeds this
LOST_GRACE = 8       # frames the object may vanish before re-searching

SEND_INTERVAL = 0.1   # seconds between motor packets (10 per second)


# ===================== CAMERA =====================
class VideoStream:
    """Reads frames in a background thread.
    PHONE_URL = 0 (int) -> laptop webcam.  PHONE_URL = "http..." -> MJPEG stream."""

    def __init__(self, source):
        self.source = source
        self.frame = None
        self.running = True
        self.status = "starting"
        target = self._run_webcam if isinstance(source, int) else self._run_mjpeg
        threading.Thread(target=target, daemon=True).start()

    def _run_webcam(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ok, img = cap.read()
            if ok:
                self.frame = img
                self.status = "ok"
            else:
                self.status = "webcam read failed"
                time.sleep(0.2)
        cap.release()

    def _run_mjpeg(self):
        while self.running:
            try:
                self.status = "connecting"
                resp = requests.get(self.source, stream=True, timeout=10, verify=False)
                if resp.status_code != 200:
                    self.status = f"HTTP {resp.status_code}"
                    time.sleep(2)
                    continue
                self.status = "ok"
                buf = b""
                for chunk in resp.iter_content(4096):
                    if not self.running:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(b"\xff\xd8")
                        if start == -1:
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end == -1:
                            buf = buf[start:]
                            break
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.frame = img
            except Exception as e:
                self.status = f"error: {type(e).__name__}"
                time.sleep(1.0)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False


# ===================== UDP COMMANDS (your udp_command_receiver.ino) =====================
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_last_send = 0.0

def _motor(in1, in2, pwm, speed):
    cmd = f"M {in1} {in2} {pwm} {int(speed)}"
    cmd_sock.sendto(cmd.encode(), (ARDUINO_IP, CMD_PORT))

def _boost(speed):
    """Push small non-zero speeds up to MIN_PWM so the motor actually turns."""
    if speed == 0:
        return 0
    sign = 1 if speed > 0 else -1
    return sign * max(abs(speed), MIN_PWM)

def send_motors(left, right):
    global _last_send
    now = time.time()
    if now - _last_send < SEND_INTERVAL:
        return
    _last_send = now
    left = max(-255, min(255, _boost(left)))
    right = max(-255, min(255, _boost(right)))
    _motor(LEFT_IN1, LEFT_IN2, LEFT_PWM, left)
    _motor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, right)

def stop_motors():
    # always send immediately (no throttle) so STOP is never skipped
    _motor(LEFT_IN1, LEFT_IN2, LEFT_PWM, 0)
    _motor(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, 0)


# ===================== WINDOWS + HSV SLIDERS =====================
def _noop(_):
    pass

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL); cv2.resizeWindow("Controls", 330, 360); cv2.moveWindow("Controls", 0, 0)
cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL); cv2.resizeWindow("Robot View", 520, 390); cv2.moveWindow("Robot View", 350, 0)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL); cv2.resizeWindow("Mask", 520, 390); cv2.moveWindow("Mask", 350, 450)

cv2.createTrackbar("H min", "Controls", 24, 179, _noop)
cv2.createTrackbar("H max", "Controls", 50, 179, _noop)
cv2.createTrackbar("S min", "Controls", 96, 255, _noop)
cv2.createTrackbar("S max", "Controls", 255, 255, _noop)
cv2.createTrackbar("V min", "Controls", 100, 255, _noop)
cv2.createTrackbar("V max", "Controls", 255, 255, _noop)
cv2.createTrackbar("Lock", "Controls", 0, 1, _noop)

def hsv_bounds():
    g = lambda n: cv2.getTrackbarPos(n, "Controls")
    return (np.array([g("H min"), g("S min"), g("V min")]),
            np.array([g("H max"), g("S max"), g("V max")]))

def find_object(frame, lower, upper):
    """Largest blob within the HSV range -> (cx, area, box, mask)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, None, mask

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < MIN_AREA:
        return None, 0, None, mask

    x, y, w, h = cv2.boundingRect(biggest)
    return x + w // 2, area, (x, y, w, h), mask


# ===================== STATE MACHINE =====================
def reached(area, box, frame_w, frame_h):
    if area >= STOP_AREA:
        return True
    if box is not None:
        x, y, bw, bh = box
        m = 5
        if (y <= m and y + bh >= frame_h - m) or (x <= m and x + bw >= frame_w - m):
            return True
    return False

def decide(cx, area, box, frame_w, frame_h):
    if cx is None:
        return -SCAN_SPEED, SCAN_SPEED, "SEARCH"

    if reached(area, box, frame_w, frame_h):
        return 0, 0, "ARRIVED"

    error = (cx - frame_w / 2) / (frame_w / 2)        # -1 (left) .. +1 (right)
    forward = FORWARD_SPEED * (1 - FORWARD_FALLOFF * min(abs(error), 1.0))
    turn = TURN_GAIN * error
    return int(forward + turn), int(forward - turn), "PURSUE"


# ===================== MAIN LOOP =====================
video = VideoStream(PHONE_URL)
print("Waiting for camera...")
t0 = time.time()
while video.read() is None:
    if time.time() - t0 > 3:
        print(f"  still waiting... status: {video.status}")
        t0 = time.time()
    time.sleep(0.1)
print("Camera connected.")

print("TUNING: adjust the sliders until only your object is white in the Mask window.")
print("Then set 'Lock' to 1 (or press 'l') to start seeking. 'q' quits.")

locked = False
locked_lower = None
locked_upper = None
arrived = False
lost = 0
last_area = 0

try:
    while True:
        frame = video.read()
        if frame is None:
            continue
        frame = frame.copy()
        h, w = frame.shape[:2]

        lock_on = cv2.getTrackbarPos("Lock", "Controls") == 1
        if lock_on and not locked:
            locked_lower, locked_upper = hsv_bounds()
            locked = True
        elif not lock_on and locked:
            locked = False
            arrived, lost, last_area = False, 0, 0
            stop_motors()

        lower, upper = (locked_lower, locked_upper) if locked else hsv_bounds()
        cx, area, box, mask = find_object(frame, lower, upper)

        left, right = 0, 0
        if not locked:
            state = "TUNING"
        else:
            if cx is not None:
                lost, last_area = 0, area
            else:
                lost += 1

            left, right, state = decide(cx, area, box, w, h)

            if state == "SEARCH" and lost <= LOST_GRACE and last_area >= STOP_AREA * 0.5:
                left, right, state = 0, 0, "ARRIVED"

            if arrived and not (cx is not None and area < STOP_AREA * 0.6):
                left, right, state = 0, 0, "ARRIVED"
            arrived = (state == "ARRIVED")

            if state == "ARRIVED":
                left, right = 0, 0
                stop_motors()
            else:
                send_motors(left, right)

        # ---- HUD ----
        if box is not None:
            x, y, bw, bh = box
            color = (0, 255, 0) if locked else (170, 170, 170)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.circle(frame, (cx, y + bh // 2), 5, (0, 0, 255), -1)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        banner = f"{'LOCKED' if locked else 'TUNING'} | {state} | area={int(area)}"
        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if locked:
            cv2.putText(frame, f"L={int(left)}  R={int(right)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Set 'Lock' to 1 (or press l) to start", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Robot View", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("l"):
            cur = cv2.getTrackbarPos("Lock", "Controls")
            cv2.setTrackbarPos("Lock", "Controls", 0 if cur == 1 else 1)

finally:
    stop_motors()
    video.stop()
    cmd_sock.close()
    cv2.destroyAllWindows()
    print("Stopped.")