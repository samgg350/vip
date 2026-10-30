# Project 1: Lime Tracker

Track a yellow lime toy with YOLO and steer the robot with PID.

| File | What it does |
|---|---|
| `collect_lime_data.py` | Captures camera frames and labels the lime automatically by its yellow color (HSV), producing a YOLO dataset in `dataset/` |
| `train_lime.py` | Trains `yolo26n.pt` on that dataset and saves `best.pt` |
| `obj_track_advpid.py` | Project 1 tracker: YOLO, PID steering, UDP motor commands |
| `check_deps.py` | Dependency check (P1.2) |

## Steps

Set up once, in this folder:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

(macOS/Linux: `source venv/bin/activate`)

```bash
pip install -r requirements.txt
```

```bash
python check_deps.py
```

1. **Settings.** Put your Arduino IP (Serial Monitor) in `config.py`. Start phonesense in a terminal and scan its QR code with the iPhone; the stream URL (`https://127.0.0.1:8080/camera/stream`) is already set.
2. **Collect data (about 5 min).**
   ```bash
   python collect_lime_data.py
   ```
   Move the sliders until only the lime is white in the Mask window. Press SPACE to record and move the lime around: near, far, left, right, and against different backgrounds. Aim for 150 or more images. Hide the lime and press N about 20 times. Press Q to finish.
3. **Train (about 10–20 min on CPU).**
   ```bash
   python train_lime.py
   ```
   This produces `best.pt`.
4. **Test detection.** Set `ROBOT_ENABLED = False` in `obj_track_advpid.py` to test without the robot moving.
   ```bash
   python obj_track_advpid.py
   ```
   Check that a box follows the lime. The `err` value should be negative when the lime is on the left and positive when it is on the right.
5. **Robot.** `ROBOT_ENABLED = True` is set. Keep `SPEED = 0`, and check that the robot turns toward the lime. If a wheel spins the wrong way, flip `LEFT_DIR` or `RIGHT_DIR`. Then raise `SPEED` to about 200.

**No `best.pt` yet?** The tracker automatically uses the pretrained `yolo26n.pt` and tracks a ball, orange or apple instead.

## Notes

- Motor commands use the same protocol as `drive_control.py` (`M in1 in2 pwm speed`, UDP port 5005), not the course's `MOTOR,l,r` on 5001.
- The robot stops whenever the lime is not visible (`STOP_WHEN_LOST`).
- Robot oscillates left and right: lower `kp` in `PID(0.7, 0.1, 0.05, ...)` toward 0.4.
