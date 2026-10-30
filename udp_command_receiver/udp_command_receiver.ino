// UDP command receiver for drive_control.py, color_seeker.py and obj_track_advpid.py
// Commands (UDP port 5005):
//   M in1 in2 pwm speed   -> motor, speed -255..255 (negative = reverse)
//   S pin angle           -> servo, angle 0..180
#include <WiFiS3.h>      // Arduino UNO R4 WiFi. Nano 33 IoT / UNO WiFi Rev2: use <WiFiNINA.h>
#include <WiFiUdp.h>
#include <Servo.h>

char ssid[] = "YOUR_WIFI_NAME";      // <-- same WiFi/hotspot as the laptop
char pass[] = "YOUR_WIFI_PASSWORD";
const int CMD_PORT = 5005;

WiFiUDP udp;
Servo servo;
int servoPin = -1;
char packet[64];

void setMotor(int in1, int in2, int pwm, int speed) {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(pwm, OUTPUT);
  speed = constrain(speed, -255, 255);
  digitalWrite(in1, speed > 0 ? HIGH : LOW);
  digitalWrite(in2, speed < 0 ? HIGH : LOW);
  analogWrite(pwm, abs(speed));
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}

  Serial.print("Connecting to ");
  Serial.println(ssid);
  while (WiFi.begin(ssid, pass) != WL_CONNECTED) {
    Serial.println("  retrying...");
    delay(2000);
  }
  while (WiFi.localIP() == IPAddress(0, 0, 0, 0)) delay(200);

  udp.begin(CMD_PORT);
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  Serial.println("Ready");
}

void loop() {
  int len = udp.parsePacket();
  if (len <= 0) return;

  len = udp.read(packet, sizeof(packet) - 1);
  packet[len] = '\0';

  int a, b, c, d;
  if (sscanf(packet, "M %d %d %d %d", &a, &b, &c, &d) == 4) {
    setMotor(a, b, c, d);
  } else if (sscanf(packet, "S %d %d", &a, &b) == 2) {
    if (servoPin != a) {
      servo.attach(a);
      servoPin = a;
    }
    servo.write(constrain(b, 0, 180));
  }
}
