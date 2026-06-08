"""
Single-script face-tracking pipeline for the eye-in-hand turret.

Runs on the Raspberry Pi with the Pi Camera mounted on the arm's end effector:

  1. Capture video from the Pi Camera (picamera2). Falls back to a USB webcam
     (cv2.VideoCapture) so the same script also runs on a laptop for testing.
  2. Detect faces with OpenCV's Haar cascade and draw boxes around them.
  3. Compute the normalized position error of the tracked face relative to the
     frame center and stream "ex,ey" to the ESP32 over serial.

The ESP32 (turret_servo_control.ino) runs the PD loop that drives the base (yaw)
and shoulder (pitch) joints, so this script only sends the error each frame.

Usage:
  - Flash turret_servo_control/turret_servo_control.ino to the ESP32.
  - The serial port auto-selects by OS (PORT_LINUX on the Pi, PORT_WINDOWS on
    Windows); edit those defaults to match your ESP32's device/COM number.
  - Run: python3 pi_face_tracking.py
  - Press ESC (in the preview window) or Ctrl+C to quit.

If a joint drives the wrong way and runs to its limit, flip INVERT_X / INVERT_Y.
"""
import os
import platform
import sys
import time

import cv2
import serial

# --- Serial link to the ESP32 (must match Serial.begin in the .ino) ---
# The port auto-selects by OS so the same script works on the Pi and on a
# Windows machine for testing. Override either default to match your setup.
PORT_LINUX = "/dev/ttyUSB0"   # Pi: /dev/ttyUSB0 or /dev/ttyACM0
PORT_WINDOWS = "COM5"         # Windows: check Device Manager for the COM number
PORT = PORT_WINDOWS if platform.system() == "Windows" else PORT_LINUX
BAUD = 115200

# --- Camera / detection settings ---
FRAME_W, FRAME_H = 640, 480   # drop to 320x240 if the Pi 3B is too slow
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE = (60, 60)

# --- Control signal shaping ---
DEADBAND = 0.03      # ignore tiny errors near center to avoid jitter
INVERT_X = False     # flip if the base drives away from the face
INVERT_Y = False     # flip if the pitch joints drive away from the face

SHOW_PREVIEW = True  # set False for headless runs

CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml",
)


class Camera:
    """Frame source: Pi Camera via picamera2, or a USB webcam as fallback."""

    def __init__(self, width, height):
        self._picam = None
        self._cap = None
        try:
            from picamera2 import Picamera2

            self._picam = Picamera2()
            config = self._picam.create_preview_configuration(
                main={"format": "RGB888", "size": (width, height)}
            )
            self._picam.configure(config)
            self._picam.start()
            time.sleep(1.0)  # let auto exposure / white balance settle
            print("Capturing from Pi Camera (picamera2).")
        except Exception as exc:  # picamera2 missing or no CSI camera
            print(f"picamera2 unavailable ({exc}); falling back to USB webcam.")
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._cap.isOpened():
                raise RuntimeError("No camera available (picamera2 or USB).")

    def read(self):
        """Return a BGR frame for OpenCV.

        picamera2's "RGB888" buffer is already in OpenCV's BGR byte order, so it
        is used directly; detection runs on grayscale either way.
        """
        if self._picam is not None:
            return self._picam.capture_array()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed.")
        return frame

    def close(self):
        if self._picam is not None:
            self._picam.stop()
        if self._cap is not None:
            self._cap.release()


def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        sys.exit(f"Failed to load Haar cascade: {CASCADE_PATH}")

    camera = Camera(FRAME_W, FRAME_H)

    try:
        esp32 = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as exc:
        camera.close()
        sys.exit(f"Could not open serial port {PORT}: {exc}")
    time.sleep(2.0)  # ESP32 resets when the serial port opens
    print(f"Connected to ESP32 on {PORT}.")

    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0

    try:
        while True:
            frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_FACE
            )

            if len(faces) > 0:
                # Track the largest face (closest / most prominent).
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                fcx, fcy = x + w / 2.0, y + h / 2.0

                # Normalized error, ~[-1, 1]; 0 means centered.
                ex = (fcx - cx) / cx
                ey = (fcy - cy) / cy
                if INVERT_X:
                    ex = -ex
                if INVERT_Y:
                    ey = -ey
                if abs(ex) < DEADBAND:
                    ex = 0.0
                if abs(ey) < DEADBAND:
                    ey = 0.0

                esp32.write(f"{ex:.3f},{ey:.3f}\n".encode())

                # Draw all detections; highlight the tracked face.
                for (bx, by, bw, bh) in faces:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 69, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(fcx), int(fcy)), 4, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "TARGET LOCKED", (20, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                # No face: send zero error so the turret holds position.
                esp32.write(b"0.000,0.000\n")
                cv2.putText(frame, "NO TARGET", (20, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # Center crosshair for visual reference.
            cv2.line(frame, (int(cx), 0), (int(cx), FRAME_H), (255, 0, 0), 1)
            cv2.line(frame, (0, int(cy)), (FRAME_W, int(cy)), (255, 0, 0), 1)

            if SHOW_PREVIEW:
                cv2.imshow("SENTRY face tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            esp32.write(b"0.000,0.000\n")  # park: stop reacting on exit
        except serial.SerialException:
            pass
        esp32.close()
        camera.close()
        cv2.destroyAllWindows()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
