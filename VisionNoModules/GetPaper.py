#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:58:11 2021

@author: alexandre
"""

import cv2
import numpy as np

# Read a image
I = cv2.imread('paperonblack.jpg')
hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# define range of white color in HSV
lower_blue = np.array([90,10,10])
upper_blue = np.array([130,255,255])

#How to define this range for white color


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# dest = cv2.cornerHarris(mask, 5, 31, 0.1)
# dest = cv2.dilate(dest, None)
  
# Reverting back to the original image,
# with optimal threshold value
# I[dest > 0.01 * dest.max()]=[0, 0, 255]
corners = np.int0(cv2.goodFeaturesToTrack(mask, 4, 0.1, 100))
polygon=[]
for i in corners:
    x,y = i.ravel()
    polygon.append([x,y])
    I = cv2.circle(I, (x,y), radius=0, color=(0, 0, 255), thickness=10)
# Bitwise-AND mask and original image
polygon=[polygon]
res = cv2.bitwise_and(I,I, mask= mask)

# First find the minX minY maxX and maxY of the polygon
minX = I.shape[1]
maxX = -1
minY = I.shape[0]
maxY = -1
print(polygon[0][0])
for point in polygon[0]:

    x = point[0]
    y = point[1]

    if x < minX:
        minX = x
    if x > maxX:
        maxX = x
    if y < minY:
        minY = y
    if y > maxY:
        maxY = y

# Go over the points in the image if thay are out side of the emclosing rectangle put zero
# if not check if thay are inside the polygon or not
cropedImage = np.zeros_like(I)
for y in range(0,I.shape[0]):
    for x in range(0, I.shape[1]):

        if x < minX or x > maxX or y < minY or y > maxY:
            continue

        if cv2.pointPolygonTest(np.asarray(polygon),(x,y),False) >= 0:
            cropedImage[y, x, 0] = I[y, x, 0]
            cropedImage[y, x, 1] = I[y, x, 1]
            cropedImage[y, x, 2] = I[y, x, 2]

# Now we can crop again just the envloping rectangle
finalImage = cropedImage[minY:maxY,minX:maxX]

cv2.imwrite('finalImage.png',finalImage)


polygonStrecth = np.float32([[0,0],[finalImage.shape[1],0],[finalImage.shape[1],finalImage.shape[0]],[0,finalImage.shape[0]]])

# Convert the polygon corrdanite to the new rectnagle
polygonForTransform = np.zeros_like(polygonStrecth)
i = 0
for point in polygon[0]:

    x = point[0]
    y = point[1]

    newX = x - minX
    newY = y - minY

    polygonForTransform[i] = [newX,newY]
    i += 1


# Find affine transform
M = cv2.getPerspectiveTransform(np.asarray(polygonForTransform).astype(np.float32), np.asarray(polygonStrecth).astype(np.float32))

# Warp one image to the other
warpedImage = cv2.warpPerspective(finalImage, M, (finalImage.shape[1], finalImage.shape[0]))
# 180 degrees
angle180 = 180
scale = 1.0
(h, w) = warpedImage.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, angle180, scale)
warpedImage = cv2.warpAffine(warpedImage, M, (w, h))

while(True):
    cv2.imshow('Paper', I)   
    # cv2.imshow('Mask', mask)
    cv2.imshow('WarpedImage', warpedImage)
    if cv2.waitKey(1) == ord("q"):
       break
   
cv2.destroyAllWindows()