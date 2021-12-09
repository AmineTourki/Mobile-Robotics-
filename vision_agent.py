#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 08:56:47 2021

@author: alexandre
"""
import cv2
import numpy as np
import pygame
import math
import time

print(cv2.__version__)

# cam = cv2.VideoCapture(0)

class Vision_Agent:
    def __init__(self):
        self.converter = 0
        # self.cam = cv2.VideoCapture(0)
        # ret, frame = cam.read() 
        self.image = cv2.imread('Parcours.png')
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.angle = 0
        self.center_robot = [0,0]
        self.parcours = np.zeros((50,50), np.uint8)
        self.parcours_2_pix=[len(self.image)/50,len(self.image[0])/50]
        self.objective_red = None
        self.objective_green = None
        self.r_in_pix = 0
        self.r_in_real = 5
        
        
        
    def read_image(self):
        self.image = cv2.imread('Parcours.png')
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
    def mask_thresh(self,image,thresh_low,thresh_high):
        lower_thresh = np.array(thresh_low)
        upper_thresh = np.array(thresh_high)
        mask_hsv = cv2.inRange(image, lower_thresh, upper_thresh)
        return mask_hsv
       
    
    
    def get_robot(self):
        
        mask_blue = self.mask_thresh(self.hsv, [90,10,10], [130,255,255])
        circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT,2,30, param1=50, param2=30 , minRadius=0,maxRadius=0)
        
        detected_circles = np.uint16(np.around(circles))
        center_points = []
        rayons=[]
        
        for (x,y,r) in detected_circles[0,:]:
            cv2.circle(self.image,(x,y),r,(0,255,0),3)
            cv2.circle(self.image,(x,y),2,(0,255,255),3)
            rayons += [r]
            center_points += [x,y]
            self.center_robot[0]+=x/len(detected_circles[0,:])
            self.center_robot[1]+=y/len(detected_circles[0,:])
        
        vec1 = 0
        vec2 = 0
        
        # print(center_points)
        if(rayons[0]>rayons[1]):
            vec1=center_points[2].astype('int64')-center_points[0].astype('int64')
            vec2=center_points[3].astype('int64')-center_points[1].astype('int64')
        else:
            vec1=center_points[0].astype('int64')-center_points[2].astype('int64')
            vec2=center_points[1].astype('int64')-center_points[3].astype('int64')

        self.r_in_pix= math.sqrt(vec1**2+vec2**2)
        self.angle = math.atan2(vec2,vec1)*180/math.pi
        # print(center_points)
        cv2.circle(self.image,(int(self.center_robot[0]),int(self.center_robot[1])),2,(0,255,200),3)

    def get_obstacles(self):
        self.parcours = np.zeros((50,50), np.uint8)
        
        mask_black = self.mask_thresh(self.hsv, [0,0,0], [10,10,10])

        Canny= cv2.Canny(mask_black,10,50)
        
        #Find my contours
        contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
        # print(contours)
        
        #Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
        for cnt in contours:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.rectangle(self.parcours,(int(x/self.parcours_2_pix[0]),int(y/self.parcours_2_pix[1])),(int((x+w)/self.parcours_2_pix[0]),int((y+h)/self.parcours_2_pix[1])),255,-1)
         
        kernel = np.ones((1,1),np.uint8) 
        self.parcours = cv2.dilate(self.parcours,kernel,iterations = 1)/255
            
    def get_objectives(self):
        # define range of objective colors in HSV
        mask_red = self.mask_thresh(self.hsv, [0,70,50], [10,255,255])
        mask_green = self.mask_thresh(self.hsv, [40,40,40], [70,255,255])
        
        Canny=cv2.Canny(mask_green,10,50)
        
        #Find my contours
        contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
        
        for cnt in contours:
            
            x,y,w,h=cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(self.image, (cX, cY), 7, (255, 255, 255), -1)
            self.objective_green = (cX,cY)
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,0,0),2)
    
        Canny=cv2.Canny(mask_red,10,50)
        
        #Find my contours
        contours =cv2.findContours(Canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    
        
        for cnt in contours:
            
            x,y,w,h=cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(self.image, (cX, cY), 7, (255, 255, 255), -1)
            self.objective_red = (cX,cY)
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,0,0),2)
            
    def get_pix_2_real(self):
        return self.r_in_real/self.r_in_pix
    
    def get_grid_2_real(self):
        an_array = np.array( self.parcours_2_pix)
        return  an_array * self.get_pix_2_real()
    
    def update(self):
        self.read_image()
        self.get_robot()
        self.get_objectives()
        self.get_obstacles()
         
        
Vision = Vision_Agent()
start = time.time()
Vision.read_image()
Vision.get_robot()
Vision.get_objectives()
Vision.get_obstacles()
# print(time.time()-start)
# print(Vision.parcours)
print(Vision.get_grid_2_real())
while(True):
    # Vision.read_image()
    cv2.imshow('Image', Vision.image)   
    cv2.imshow('Parcours', Vision.parcours)   
    if cv2.waitKey(1) == ord("q"):
       break
   
cv2.destroyAllWindows()