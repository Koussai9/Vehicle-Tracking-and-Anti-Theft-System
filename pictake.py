import cv2
import os
import time
from gpiozero import Button

name = 'test'  # replace with your name
url = 'http://20.20.19.105:81/stream'
cam = cv2.VideoCapture(url)

cv2.namedWindow("press space or button to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space or button to take a photo", 500, 300)

img_counter = 0

folder_path = "dataset/" + name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print("Folder '{}' created.".format(folder_path))

button_pin = 27  
button = Button(button_pin)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("press space or button to take a photo", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32 or button.is_pressed:
        # SPACE pressed or button is pressed
        img_name = "dataset/" + name + "/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        time.sleep(1)

cam.release()
cv2.destroyAllWindows()
