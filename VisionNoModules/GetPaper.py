#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:58:11 2021

@author: alexandre
"""

import cv2
import numpy as np

# Read a image
I = cv2.imread('parcours2.jpeg')

scale_percent = 30 # percent of original size
width = int(I.shape[1] * scale_percent / 100)
height = int(I.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
I = cv2.resize(I, dim, interpolation = cv2.INTER_AREA)
hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
# define range of white color in HSV
lower_blue = np.array([0,0,50])
upper_blue = np.array([255,255,255])

#How to define this range for white color


# Threshold the HSV image to get only blue colors

mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask_bg = np.zeros((len(mask) + 1000, len(mask[0]) + 1000),np.uint8)
mask_bg[500:len(mask_bg)-500,500:len(mask_bg[0])-500]=mask
kernel = np.ones((3,3),np.uint8)
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((50,50),np.uint8)
mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel)
kernel = np.ones((150,150),np.uint8)

mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
mask = mask_bg[500:len(mask_bg)-500,500:len(mask_bg[0])-500]

corners = np.int0(cv2.goodFeaturesToTrack(mask_bg, 4, 0.1, 100))
polygon=[]
for i in corners:
    x,y = i.ravel()
    polygon.append([x-500,y-500])

polygon=[polygon]
res = cv2.bitwise_and(I,I, mask= mask)

# First find the minX minY maxX and maxY of the polygon
minX = I.shape[1]
maxX = -1
minY = I.shape[0]
maxY = -1
print(polygon[0])
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

register = polygon[0][1]
polygon[0][1] = polygon[0][0]
polygon[0][0] = register
# Go over the points in the image if thay are out side of the emclosing rectangle put zero
# if not check if thay are inside the polygon or not
cropedImage = np.zeros_like(I)
for y in range(0,I.shape[0]):
    for x in range(0, I.shape[1]):

        if x < minX or x > maxX or y < minY or y > maxY:
            continue

        if cv2.pointPolygonTest(np.asarray(polygon[0]),(x,y),False) >= 0:
            cropedImage[y, x, 0] = I[y, x, 0]
            cropedImage[y, x, 1] = I[y, x, 1]
            cropedImage[y, x, 2] = I[y, x, 2]

# Now we can crop again just the envloping rectangle
finalImage = cropedImage[minY:maxY,minX:maxX]



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
cv2.imwrite('finalImage.png',warpedImage)

while(True):
    cv2.imshow('Paper', I)   
    cv2.imshow('Mask', mask)
    cv2.imshow('WarpedImage', cropedImage)
    if cv2.waitKey(1) == ord("q"):
        break
   
cv2.destroyAllWindows()