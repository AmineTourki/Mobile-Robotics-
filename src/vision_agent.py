#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 08:56:47 2021
@author: alexandre
"""
import cv2
import numpy as np
import math
import time
from shapely.geometry import Polygon

GRID_SIZE = 40

class Vision_Agent:
    def __init__(self):
        self.converter = 0
        self.cam = cv2.VideoCapture(1)
        ret, frame = self.cam.read()
        self.image = frame
        self.resize()
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.get_first_image_cam(self.image)
        # self.image = cv2.flip(self.image, 1)
        # self.image = cv2.flip(self.image, 0)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.path = []
        self.angle = 0
        self.center_robot = [0, 0]
        self.parcours = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
        self.parcours_2_pix = [len(self.image[0]) / GRID_SIZE, len(self.image) / GRID_SIZE]
        self.objective_red = None
        self.objective_green = None
        self.r_in_pix = 0
        self.r_in_real = 6

    def resize(self):
        scale_percent = 30  # percent of original size
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)

    def read_image(self):
        try:
            ret, frame = self.cam.read()
            self.image = frame
            self.resize()
        except:
            print("Camera not found")
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.get_image_cam(self.image)
        # self.image = cv2.flip(self.image, 1)
        # self.image = cv2.flip(self.image, 0)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def mask_thresh(self, image, thresh_low, thresh_high):
        lower_thresh = np.array(thresh_low)
        upper_thresh = np.array(thresh_high)
        mask_hsv = cv2.inRange(image, lower_thresh, upper_thresh)
        return mask_hsv

    def redefine_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def print_path(self):
        for node in self.path:
            cv2.circle(self.image, (int(node[0] * self.parcours_2_pix[0]), int(node[1] * self.parcours_2_pix[1])), 1,
                       (255, 0, 0), 2)

    def get_robot(self):

        mask_blue = self.mask_thresh(self.hsv, [100, 87, 65], [164, 255, 180])
        kernel = np.ones((3, 3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        try:
            circles = cv2.HoughCircles(mask_blue, cv2.HOUGH_GRADIENT, 2, 10, param1=30, param2=10, minRadius=4,
                                       maxRadius=7)
        except:
            return False

        detected_circles = np.uint16(np.around(circles))
        center_points = []
        rayons = []
        self.center_robot = [0, 0]
        if len(detected_circles[0]) == 2:
            for (x, y, r) in detected_circles[0, :]:
                cv2.circle(self.image, (x, y), r, (0, 255, 0), 3)
                rayons += [r]
                center_points += [x, y]
                self.center_robot[0] += x / len(detected_circles[0, :])
                self.center_robot[1] += y / len(detected_circles[0, :])

            vec1 = 0
            vec2 = 0

            if (rayons[0] > rayons[1]):
                vec1 = center_points[2].astype('int64') - center_points[0].astype('int64')
                vec2 = center_points[3].astype('int64') - center_points[1].astype('int64')
            else:
                vec1 = center_points[0].astype('int64') - center_points[2].astype('int64')
                vec2 = center_points[1].astype('int64') - center_points[3].astype('int64')

            self.r_in_pix = math.sqrt(vec1 ** 2 + vec2 ** 2)
            self.angle = math.atan2(vec2, vec1) * 180 / math.pi
            cv2.circle(self.image, (int(self.center_robot[0]), int(self.center_robot[1])), 2, (0, 255, 200), 3)
            self.center_robot = [float(self.center_robot[0] / self.parcours_2_pix[0]),
                                 float(self.center_robot[1] / self.parcours_2_pix[1])]
            # print("pos dans get robot", self.center_robot)
            return True
        else:
            return False

    def get_obstacles(self):
        self.parcours = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)

        mask_black = self.mask_thresh(self.hsv, [0, 0, 0], [180, 255, 50])

        kernel = np.ones((14, 14), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        Canny = cv2.Canny(mask_black, 10, 50)

        # Find my contours
        contours = cv2.findContours(Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(contours)

        # Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(self.parcours, (int(x / self.parcours_2_pix[0]), int(y / self.parcours_2_pix[1])),
                          (int((x + w) / self.parcours_2_pix[0]), int((y + h) / self.parcours_2_pix[1])), 255, -1)
        kernel = np.ones((5, 5), np.uint8)
        self.parcours = np.transpose(cv2.dilate(self.parcours, kernel, iterations=2) / 255)

    def get_objectives(self):
        # define range of objective colors in HSV
        mask_red = self.mask_thresh(self.hsv, [150, 70, 50], [190, 255, 255])
        mask_green = self.mask_thresh(self.hsv, [40, 40, 40], [100, 255, 255])
        kernel = np.ones((13, 13), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = self.redefine_mask(mask_green)
        mask_red = self.redefine_mask(mask_red)

        Canny = cv2.Canny(mask_green, 10, 50)
        # Find my contours
        contours = cv2.findContours(Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cv2.circle(self.image, (int(x + w / 2), int(y + h / 2)), 7, (255, 255, 255), -1)

            self.objective_green = (x + w / 2, y + h / 2)
            self.objective_green = [math.floor(float(self.objective_green[0] / self.parcours_2_pix[0])),
                                    math.floor(float(self.objective_green[1] / self.parcours_2_pix[1]))]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 0), 2)

        Canny = cv2.Canny(mask_red, 10, 50)

        # Find my contours
        contours = cv2.findContours(Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(contours)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.circle(self.image, (int(x + w / 2), int(y + h / 2)), 7, (255, 255, 255), -1)
            self.objective_red = (x + w / 2, y + h / 2)
            self.objective_red = [math.floor(float(self.objective_red[0] / self.parcours_2_pix[0])),
                                  math.floor(float(self.objective_red[1] / self.parcours_2_pix[1]))]

            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 0), 2)

    def get_pix_2_real(self):
        return self.r_in_real / self.r_in_pix

    def get_grid_2_real(self):
        an_array = np.array(self.parcours_2_pix)
        return an_array * self.get_pix_2_real()

    def update(self):
        self.get_robot()
        self.get_objectives()
        self.get_obstacles()

    def get_first_image_cam(self, image):
        mask = self.mask_thresh(self.hsv, [0, 0, 40], [255, 255, 255])
        mask_bg = np.zeros((len(mask) + 1000, len(mask[0]) + 1000), np.uint8)
        mask_bg[500:len(mask_bg) - 500, 500:len(mask_bg[0]) - 500] = mask
        kernel = np.ones((3, 3), np.uint8)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((50, 50), np.uint8)
        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((150, 150), np.uint8)

        mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel)
        mask = mask_bg[500:len(mask_bg) - 500, 500:len(mask_bg[0]) - 500]

        corners = np.int0(cv2.goodFeaturesToTrack(mask_bg, 4, 0.1, 100))
        polygon = []
        for i in corners:
            x, y = i.ravel()
            polygon.append([x - 500, y - 500])

        polygon = [polygon]
        res = cv2.bitwise_and(image, image, mask=mask)

        # First find the minX minY maxX and maxY of the polygon
        self.minX = image.shape[1]
        self.maxX = -1
        self.minY = image.shape[0]
        self.maxY = -1
        for point in polygon[0]:

            x = point[0]
            y = point[1]

            if x < self.minX:
                self.minX = x
            if x > self.maxX:
                self.maxX = x
            if y < self.minY:
                self.minY = y
            if y > self.maxY:
                self.maxY = y
        center_x = 0
        center_y = 0
        for i in range(len(polygon[0])):
            x = polygon[0][i][0]
            y = polygon[0][i][1]
            center_x += x / 4
            center_y += y / 4
            if (x <= polygon[0][0][0] - 10 and y <= polygon[0][0][1] - 10):
                register = polygon[0][0]
                polygon[0][0] = polygon[0][i]
                # print("yo1")
                polygon[0][i] = register
            if (x <= polygon[0][1][0] - 10 and y >= polygon[0][1][1] - 10):
                register = polygon[0][1]
                polygon[0][1] = polygon[0][i]
                polygon[0][i] = register
                # print("yo2")

            if (x >= polygon[0][2][0] - 10 and y >= polygon[0][2][1] - 10):
                register = polygon[0][2]
                polygon[0][2] = polygon[0][i]
                polygon[0][i] = register
                # print("yo3")

            if (x >= polygon[0][3][0] - 10 and y <= polygon[0][3][1] - 10):
                register = polygon[0][3]
                polygon[0][3] = polygon[0][i]
                polygon[0][i] = register
                # print("yo4")

        if not Polygon(np.array(polygon[0])).is_valid:
            # print("you")
            register = polygon[0][3]
            polygon[0][3] = polygon[0][0]
            polygon[0][0] = register

        # Go over the points in the image if they are out side of the enclosing rectangle put zero
        # if not check if they are inside the polygon or not
        self.polygon = polygon

        cropedImage = np.zeros_like(image)
        mask_im = np.zeros((len(image), len(image[0])), np.uint8)
        for y in range(0, image.shape[0]):
            for x in range(0, image.shape[1]):

                if x < self.minX or x > self.maxX or y < self.minY or y > self.maxY:
                    continue

                # if cv2.pointPolygonTest(np.asarray(polygon[0]), (x, y), False) >= 0:
                cropedImage[y, x, 0] = image[y, x, 0]
                cropedImage[y, x, 1] = image[y, x, 1]
                cropedImage[y, x, 2] = image[y, x, 2]
                mask_im[y, x] = 1

        # Now we can crop again just the enveloping rectangle
        self.mask_im = mask_im
        finalImage = cropedImage[self.minY:self.maxY, self.minX:self.maxX]

        polygonStrecth = np.float32(
            [[0, 0], [finalImage.shape[1], 0], [finalImage.shape[1], finalImage.shape[0]], [0, finalImage.shape[0]]])

        # Convert the polygon coordinate to the new rectangle
        polygonForTransform = np.zeros_like(polygonStrecth)
        i = 0
        for point in polygon[0]:
            x = point[0]
            y = point[1]

            newX = x - self.minX
            newY = y - self.minY

            polygonForTransform[i] = [newX, newY]
            i += 1

        # Find affine transform
        M = cv2.getPerspectiveTransform(np.asarray(polygonForTransform).astype(np.float32),
                                        np.asarray(polygonStrecth).astype(np.float32))
        self.M = M

        # Warp one image to the other
        warpedImage = cv2.warpPerspective(finalImage, M, (finalImage.shape[1], finalImage.shape[0]))
        # 180 degrees
        angle180 = 180
        scale = 1.0
        (h, w) = warpedImage.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        # warpedImage = cv2.warpAffine(warpedImage, M, (w, h))
        self.image = finalImage

    def get_image_cam(self, image):

        polygon = self.polygon
        cropedImage = np.zeros_like(image)

        f = time.time()

        # get first masked value (foreground)
        cropedImage = cv2.bitwise_or(image, image, mask=self.mask_im)

        # Now we can crop again just the enveloping rectangle
        finalImage = cropedImage[self.minY:self.maxY, self.minX:self.maxX]

        M = self.M
        # Warp one image to the other
        warpedImage = cv2.warpPerspective(finalImage, M, (finalImage.shape[1], finalImage.shape[0]))
        # 180 degrees
        angle180 = 180
        scale = 1.0
        (h, w) = warpedImage.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        warpedImage = cv2.warpAffine(warpedImage, M, (w, h))
        self.image = finalImage

    def read_save_image(self, title):
        self.read_image()
        cv2.imwrite(title, self.image)

# Vision=Vision_Agent()

# Vision.read_image()
# Vision.get_robot()
# Vision.get_objectives()
# Vision.get_obstacles()

# while (True):
#    # Vision.read_image()
#
#     cv2.imshow('Image', Vision.image)
#     cv2.imshow('Parcours', Vision.parcours)
#     if cv2.waitKey(1) == ord("q"):
#         break
# Vision.cam.release()
# cv2.destroyAllWindows()
