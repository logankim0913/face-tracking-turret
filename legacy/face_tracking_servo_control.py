""" 
Script's function: face recognition based on video streaming with laptop camera and updating servomotor's position based on the output;

In order for the script to work, you first have to connect the Arduino to your computer.
Adjust the path of CascadeClassifier and the port number of Arduino connection when necessary.
Load the turret_servo_control.ino code to your Arduino then execute this script.
"""
import cv2 as cv
import serial
import time

# cascades for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# open the window for video streaming
cv.namedWindow("Face detection")
cam = cv.VideoCapture(0)

# getting python-arduino connection
arduino = serial.Serial("COM5", 115200)   # Edit the COM port number
time.sleep(2)
print("Connection to Arduino...")

while True:
    success, frame = cam.read()
    if not success:
        raise RuntimeError("Camera failed.")
    
    hs, ws = frame.shape[:2]    # camera frame size
    cx, cy = (ws / 2), (hs / 2) # center of camera frame
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_frame, 1.3, 5, 10)

    for (x, y, w, h) in face:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 69, 255), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]  # slicing off the face from the image(grayscale)
        roi_color = frame[y:y + h, x:x + w]  # slicing off the face from the image(color)
        fcx, fcy = x + w/2, y + h/2  # center of frame
        
        # error calculation, normalized
        ex, ey = ((fcx - cx) / cx), ((fcy - cy) / cy)
        if abs(ex) < 0.03: ex = 0
        if abs(ey) < 0.03: ey = 0
        
        error = f"{ex:.3f},{ey:.3f}\n"
        arduino.write(error.encode())
        print("Face coordinates: ({0:f}, {1:f})".format(fcx, fcy))

    cv.imshow("Face detection", frame)
    
    # # Serial monitoring for debugging purposes
    # last_print = 0
    
    # if arduino.in_waiting > 0:
    #     try:
    #         line = arduino.readline()
    #         now = time.time()
    #         if now - last_print > 0.2:
    #             print(line.decode('utf-8')) # Decode bytes to string and print
    #             last_print = now
    #     except:
    #         print("Error reading from serial port")
    #         break
    
    # Exit condition
    k = cv.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

arduino.close()
cam.release()
cv.destroyAllWindows()
