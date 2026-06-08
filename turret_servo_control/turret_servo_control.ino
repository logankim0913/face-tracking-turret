#include <ESP32Servo.h>

/*
 * 2-axis (pan/tilt) face-tracking turret controller on a 4-DOF arm.
 *
 * Receives the normalized face-position error "ex,ey\n" from the host (the Pi
 * running pi_face_tracking.py) over serial, where ex/ey are in roughly [-1, 1]
 * with 0 meaning the face is centered in the camera frame.
 *
 * Because the camera is mounted on the arm (eye-in-hand), reducing the on-screen
 * error is equivalent to aiming the turret at the face, so a per-joint PD loop on
 * the incremental angle is a stable closed loop. Centering a face only needs two
 * degrees of freedom, so only two joints are driven:
 *   - base     yaw   tracks horizontal error ex
 *   - shoulder pitch tracks vertical error ey
 * The elbow and wrist are parked (held) at their center angle so the arm stays
 * rigid without contributing redundant pitch motion.
 */

Servo base;      // yaw joint   - corrects horizontal error (ex)
Servo shoulder;  // pitch joint - corrects vertical error (ey)
Servo elbow;     // locked at center
Servo wrist;     // locked at center

// GPIO pins for each servo signal line.
// Recommended PWM-capable ESP32 pins: 2,4,12-19,21-23,25-27,32-33
const int PIN_BASE = 12;
const int PIN_SHOULDER = 13;
const int PIN_ELBOW = 14;
const int PIN_WRIST = 27;

// Per-joint angle limits and center (start) position.
const float BASE_MIN = 0;     const float BASE_MAX = 120;     const float BASE_CENTER = (BASE_MAX + BASE_MIN) / 2;
const float SHOULDER_MIN = 0; const float SHOULDER_MAX = 120; const float SHOULDER_CENTER = (SHOULDER_MAX + SHOULDER_MIN) / 2;
const float ELBOW_CENTER = 60;   // parked angle (held, not tracked)
const float WRIST_CENTER = 60;   // parked angle (held, not tracked)

// Current commanded angle for the two driven joints, starting centered.
float baseAngle = BASE_CENTER;
float shoulderAngle = SHOULDER_CENTER;

// PD control gains. Base reacts to ex (yaw); shoulder reacts to ey (pitch).
// Shoulder is the sole pitch joint, so it carries a higher gain than when the
// pitch error was previously shared across several joints.
const float P_BASE = 10.0;    const float D_BASE = 0.1;
const float P_SHOULDER = 5.0; const float D_SHOULDER = 0.1;

const float MAX_STEP = 2.5;   // max degrees changed per update (rate limit)
const float DEADBAND = 0.05;  // ignore tiny errors near center to avoid jitter

float ex = 0, ey = 0;             // latest normalized errors
float ex_prev = 0, ey_prev = 0;   // previous errors for the derivative term

// Advance one joint's angle by a rate-limited PD step, then clamp to its limits.
// Within the deadband the joint holds position.
float updateJoint(float angle, float gainP, float gainD, float err, float dErr,
                  float lo, float hi) {
  if (fabs(err) >= DEADBAND) {
    float step = gainP * err + gainD * dErr;
    step = constrain(step, -MAX_STEP, MAX_STEP);
    angle += step;
  }
  return constrain(angle, lo, hi);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  // ESP32Servo needs dedicated LEDC timers; allocate one per servo.
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  base.setPeriodHertz(50);
  shoulder.setPeriodHertz(50);
  elbow.setPeriodHertz(50);
  wrist.setPeriodHertz(50);

  base.attach(PIN_BASE, 500, 2400);
  shoulder.attach(PIN_SHOULDER, 500, 2400);
  elbow.attach(PIN_ELBOW, 500, 2400);
  wrist.attach(PIN_WRIST, 500, 2400);

  base.write(baseAngle);
  shoulder.write(shoulderAngle);

  // Park the unused joints at center and leave them holding that position.
  elbow.write(ELBOW_CENTER);
  wrist.write(WRIST_CENTER);

  delay(1000);
  Serial.println("Turret initialization ready.");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  // Expect one "ex,ey" pair per line.
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;
  }

  int comma = line.indexOf(',');
  if (comma < 0) {
    return;  // malformed line, ignore
  }

  ex = line.substring(0, comma).toFloat();
  ey = line.substring(comma + 1).toFloat();

  float dex = ex - ex_prev;
  float dey = ey - ey_prev;

  // Base handles yaw (horizontal); shoulder handles pitch (vertical).
  baseAngle     = updateJoint(baseAngle,     P_BASE,     D_BASE,     ex, dex, BASE_MIN,     BASE_MAX);
  shoulderAngle = updateJoint(shoulderAngle, P_SHOULDER, D_SHOULDER, ey, dey, SHOULDER_MIN, SHOULDER_MAX);

  base.write(baseAngle);
  shoulder.write(shoulderAngle);

  ex_prev = ex;
  ey_prev = ey;
}
