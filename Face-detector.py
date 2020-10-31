# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 07:41:21 2020

@author: Ashish Aman
"""

import cv2

from random import randrange


# load some pre-trained data on face fronals from opencv (haar cascade algorithm)
# calling opencv lib and from there --> CascadeClassifier() fun --> Callifier are fancy word for detectors  --> classifier can classify something as face  --> cascade is the algorithm
# we'll pass the front facing training data and create a classifier for this --> and then classifier will be able to detect the front faces.
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect face in --> by calling img read func. from open cv
img = cv2.imread('A4.jpg')  # img is just an 2D array with pixels in no.

# the haar cascade algo only takes black & white img so we need to convert the image in B&W
# convert to grayscale
# we are calling convert col fun. here --> we give 2 args --> src img, and what type of conversion we want (here to gray) BGR --> (main color - Red blue green) --> BGR2GRAY conversion
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# we want to train the algo now --> but that has already happed opencv was able to already do that

# now we need to plug that gray image into our algo that was given by opencv --> (xml file) --> from there we'll be able to detect faces

# Detect Faces
# from trained_face_data classifier we call func. detectMultiScale() --> all it means from face classifier that we trained on we want to detect all the faces with multi scale thing --> no matter scale of face (smaller/bigger)
# detected objects are returned as a list of rectangles --> coordinates of green rectangles.
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# returns upper left and bottom right coordinates
print(face_coordinates)

# Draw rectangle around the faces --> (x,y) upper left coordinate, (x+w, y+h) bottom right coordinate , (w,h) is width and height of rectangle
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  --> 2 is thickness of rectangle, (255) in mid is green, 0 in last is red, 0 in 1st is blue
for (x, y, w, h) in face_coordinates:  # face_coordinate returns the list of the coordinates of faces
    cv2.rectangle(img, (x, y), (x+w, y+h),
                  (randrange(256), randrange(256), randrange(256)), 2)
# cv2.rectangle(img, (156, 199), (156+650, 199+650), (0, 255, 0), 2)


# show the img with faces
cv2.imshow('Face Detector', img)

# pauses the execution of code --> untill a key is pressed to continue the execution of code
cv2.waitKey()

print('Code Completed!')
