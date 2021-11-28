#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:00:24 2021

@author: alexandre
"""

import cv2
import numpy as np
import pygame
import math

def angle_of_vector(x, y):
    #return math.degrees(math.atan2(-y, x))            # 1: with math.atan
    return pygame.math.Vector2(x, y).angle_to((1, 0))  # 2: with pygame.math.Vector2.angle_to
    
def angle_of_line(x1, y1, x2, y2):
    #return math.degrees(math.atan2(-y1-y2, x2-x1))    # 1: math.atan
    return angle_of_vector(x2-x1, y2-y1)               # 2: pygame.math.Vector2.angle_to

# Read a image
I = cv2.imread('Parcours.png')
parcours = np.zeros((len(I),len(I[0])), np.uint8)
hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

#Get Start

# define range of blue color in HSV
lower_blue = np.array([90,10,10])
upper_blue = np.array([130,255,255])

mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
 
circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT,2,30, param1=50, param2=30 , minRadius=0,maxRadius=0)
# print(circles)

detected_circles = np.uint16(np.around(circles))
center_robot = [0,0]
center_points = []
rayons=[]

for (x,y,r) in detected_circles[0,:]:
    cv2.circle(I,(x,y),r,(0,255,0),3)
    cv2.circle(I,(x,y),2,(0,255,255),3)
    rayons += [r]
    center_points += [x,y]
    center_robot[0]+=x/len(detected_circles[0,:])
    center_robot[1]+=y/len(detected_circles[0,:])

vec1 = 0
vec2 = 0
# print(center_points)
if(rayons[1]>rayons[0]):
    vec1=center_points[2].astype('int64')-center_points[0].astype('int64')
    vec2=center_points[3].astype('int64')-center_points[1].astype('int64')
else:
    vec1=center_points[0].astype('int64')-center_points[2].astype('int64')
    vec2=center_points[1].astype('int64')-center_points[3].astype('int64')
    

angle = math.atan(vec2/vec1)*360/math.pi
if (vec1<0):
    angle+=180
print(angle)
# print(center_points)
cv2.circle(I,(int(center_robot[0]),int(center_robot[1])),2,(0,255,200),3)

#Get Obstacle

# define range of black color in HSV
lower_black = np.array([0,0,0])
upper_black = np.array([10,10,10])

mask_black = cv2.inRange(hsv, lower_black, upper_black)

Canny=cv2.Canny(mask_black,10,50)

#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
# print(contours)
#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
for cnt in contours:
    x,y,w,h=cv2.boundingRect(cnt)
    cv2.rectangle(I,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.rectangle(parcours,(x,y),(x+w,y+h),255,-1)

    


#Get Objectives

# define range of objective colors in HSV
lower_green = np.array([40,40,40])
upper_green = np.array([70,255,255])
lower_red = np.array([0,70,50])
upper_red = np.array([10,255,255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)

mask_green = cv2.inRange(hsv, lower_green, upper_green)

mask_obj = mask_green + mask_red

Canny=cv2.Canny(mask_obj,10,50)

#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
# print(contours)
#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
objectives = []

for cnt in contours:
    
    x,y,w,h=cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(I, (cX, cY), 7, (255, 255, 255), -1)
    
    objectives += (cX,cY)
    
    cv2.rectangle(I,(x,y),(x+w,y+h),(0,0,0),2)

print(objectives)




                      
while(True):
    cv2.imshow('Image', I)   
    cv2.imshow('Parcours', parcours)   
    if cv2.waitKey(1) == ord("q"):
       break
   
cv2.destroyAllWindows()