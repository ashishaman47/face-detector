import cv2

from random import randrange

# load some pre-trained data on face fronals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# To capture video from webcam. --> if arg is 0 then reads from webcam
webcam = cv2.VideoCapture(0)

# Adding the video file --> if you give name it reads from file path
# webcam = cv2.VideoCapture('Dangal.mkv')

# Iterate over the frames in the video forever untill video/webcam ends
while True:
    # Read the current frame --> returns 2 thing --> boolean value whether the frame is successfully read or not, and the img i.e frame that has been read correctly
    successful_frame_read, frame = webcam.read()

    # Convering frame captured to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in form of coordinates (x,y,w,h)
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangle around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 2)

    # Show the captured frame --> not the converted grayscale image
    cv2.imshow('Face Detector >>>', frame)
    # wait for 1 milli sec before moving to next frame --> so we don't need to press any key
    # at the same time listen to any key pressed
    key = cv2.waitKey(1)  # if you don't press the key it's null

    # STOP if Q key is pressed (ASCCI key of Q is used)
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()


print('Code Completed!')
