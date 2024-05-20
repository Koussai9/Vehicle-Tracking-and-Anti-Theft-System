import cv2
import numpy as np
import RPi.GPIO as GPIO
import face_recognition
import pickle
import time
import os
import pyrebase




# Firebase configuration
firebaseConfig={
    "apiKey": "",
    "authDomain": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "databaseURL": ""
    }

# Initialize Firebase with pyrebase
firebase = pyrebase.initialize_app({
    "apiKey": firebaseConfig["apiKey"],
    "authDomain": firebaseConfig["authDomain"],
    "databaseURL": firebaseConfig["databaseURL"],
    "storageBucket": firebaseConfig["storageBucket"]
})

storage = firebase.storage()
 
# GPIO pin to which the vibration sensor is connected
vibration_pin = 17
RelayPin = 22
# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(RelayPin, GPIO.OUT) # set the RealyPin to OUTPUT mode
GPIO.output(RelayPin, GPIO.LOW) # make RelayPin output LOW level 
GPIO.setmode(GPIO.BCM)
GPIO.setup(vibration_pin, GPIO.IN)
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT)

GPIO.output(12, GPIO.HIGH)
 
# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
 
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
 
# Folder path to save unknown faces
unknown_faces_folder = "/home/admin/Desktop/GPS_TRACKER/unknown_faces"
 
# Create the folder if it doesn't exist
os.makedirs(unknown_faces_folder, exist_ok=True)
 
# Camera URL
url = 'http://20.20.19.105:81/stream'
 
while True:
    # Open the video stream from the URL
    cap = cv2.VideoCapture(url)
    time.sleep(2.0)  # Allow the camera to warm up
 
    try:
        while True:
            # Read the state of the vibration sensor
            is_vibration_detected = GPIO.input(vibration_pin)
 
            if is_vibration_detected == GPIO.HIGH:
                print("Vibration detected!")
 
                # grab the frame from the video stream
                ret, frame = cap.read()
 
                if not ret:
                    print("Failed to capture video frame")
                    break
 
                # Convert the frame to RGB for face recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
                # Detect the face boxes
                boxes = face_recognition.face_locations(rgb_frame)
                names = []
 
                # loop over the facial embeddings
                for (top, right, bottom, left) in boxes:
                    # draw the predicted face name on the image - color is in BGR
                    cv2.rectangle(frame, (left, top), (right, bottom),
                                  (0, 255, 225), 2)
 
                    # Extract the face encoding for the current face
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
 
                    # Check if the face is unknown
                    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
                    if not any(matches):
                        # Save the frame as an image for unknown faces
                        timestamp = time.strftime("%Y%m%d%H%M%S")
                        filename = os.path.join(unknown_faces_folder, f"unknown_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
 
                # Display the image to our screen
                cv2.imshow("Facial Recognition with Vibration Sensor", frame)
 
                # Wait for a short duration before checking again
                time.sleep(1)
 
                # Continue scanning for one minute (60 seconds)
                end_time = time.time() + 30
                while time.time() < end_time:
                    # grab the frame from the video stream
                    ret, frame = cap.read()
 
                    if not ret:
                        print("Failed to capture video frame. Retrying...")
                        break
 
                    # Convert the frame to RGB for face recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
                    # Detect the face boxes
                    boxes = face_recognition.face_locations(rgb_frame)
                    names = []
 
                    # loop over the facial embeddings
                    for (top, right, bottom, left) in boxes:
                        # draw the predicted face name on the image - color is in BGR
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                      (0, 255, 225), 2)
 
                        # Extract the face encoding for the current face
                        face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
 
                        # Check if the face is unknown
                        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
                        if not any(matches):
                            # Save the frame as an image for unknown faces
                            timestamp = time.strftime("%Y %m %d %H: %M: %S")
                            filename = os.path.join(unknown_faces_folder, f"unknown_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)
							# Upload the image to Firebase Storage
                            storage.child(f"unknown_faces/{os.path.basename(filename)}").put(filename)
                            GPIO.output(RelayPin, GPIO.HIGH)
                            GPIO.output(12, GPIO.LOW)
                            time.sleep(35)
                            GPIO.output(12, GPIO.HIGH)
                    # Display the image to our screen
                    cv2.imshow("Facial Recognition with Vibration Sensor", frame)
 
                    # Wait for a short duration before checking again
                    time.sleep(1)
 
            else:
                print("No vibration")
 
                # Wait for a short duration before checking again
                time.sleep(1)
 
    except KeyboardInterrupt:
        # Clean up GPIO on program exit
        GPIO.cleanup()
 
    finally:
        # Release the video stream
        cap.release()
 
        # do a bit of cleanup
        cv2.destroyAllWindows()
