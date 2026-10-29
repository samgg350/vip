import socket
import tkinter as tk

ARDUINO_IP = "192.168.1.217"   # <-- put your Arduino IP here
ARDUINO_PORT = 5005

# Servo pin
SERVO_PIN = 10

# Left motor pins (MOTOR #1 on the shield)
LEFT_IN1 = 4
LEFT_IN2 = 5
LEFT_PWM = 6

# Right motor pins (MOTOR #2 on the shield)
RIGHT_IN1 = 12  
RIGHT_IN2 = 13
RIGHT_PWM = 11

MOTOR_SPEED = 170

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(command):
    print("Sending:", command)
    sock.sendto(command.encode(), (ARDUINO_IP, ARDUINO_PORT))


def servo_changed(value):
    send(f"S {SERVO_PIN} {int(value)}")


def left_motor(speed):
    send(f"M {LEFT_IN1} {LEFT_IN2} {LEFT_PWM} {speed}")


def right_motor(speed):
    send(f"M {RIGHT_IN1} {RIGHT_IN2} {RIGHT_PWM} {speed}")


def forward():
    left_motor(MOTOR_SPEED)
    right_motor(MOTOR_SPEED)


def backward():
    left_motor(-MOTOR_SPEED)
    right_motor(-MOTOR_SPEED)


def turn_left():
    left_motor(-MOTOR_SPEED)
    right_motor(MOTOR_SPEED)


def turn_right():
    left_motor(MOTOR_SPEED)
    right_motor(-MOTOR_SPEED)


def stop_motors():
    left_motor(0)
    right_motor(0)


root = tk.Tk()
root.title("Servo and Motor Control")
root.geometry("340x320")

servo_slider = tk.Scale(root, from_=0, to=180, orient="horizontal",
                        label="Servo Angle", command=servo_changed)
servo_slider.pack(fill="x", padx=20, pady=15)
servo_slider.set(90)

tk.Label(root, text="DC Motor Control").pack(pady=5)
tk.Button(root, text="Forward", width=15, command=forward).pack(pady=5)

button_row = tk.Frame(root)
button_row.pack(pady=5)
tk.Button(button_row, text="Turn Left", width=12, command=turn_left).grid(row=0, column=0, padx=5)
tk.Button(button_row, text="Turn Right", width=12, command=turn_right).grid(row=0, column=1, padx=5)

tk.Button(root, text="Backward", width=15, command=backward).pack(pady=5)
tk.Button(root, text="STOP", width=15, command=stop_motors).pack(pady=10)

root.mainloop()