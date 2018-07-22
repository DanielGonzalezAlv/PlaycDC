#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:18:57 2018

@author: daniel
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import imutils

### Path to use
path_img = "/home/daniel/Studium/Uni-Heidelberg/Vorlesungen-Projects/ORIU/Project/Images/Post-Processed/"


img = cv2.imread(path_img + "4.JPG")
frame = img

frame = cv2.flip(frame, 1)
frame = imutils.resize(frame, width=650, height=950)
height, width, _ = img.shape



frame = cv2.flip(frame, 1)
image = frame.copy() #copy frame so that we don't get funky contour problems when drawing contours directly onto the frame.

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray, 11, 17, 17) 
edges = imutils.auto_canny(gray)

plt.imshow(edges)
plt.show()

#find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
_, cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]
screenCnt = None
#
## loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
 
    # if our approximated contour has four points, then
    # we can assume that we have found our card
    if len(approx) == 4:
        screenCnt = approx
        break

a =cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

plt.imshow(a)
plt.show()


pts = screenCnt

mask = np.zeros((image.shape[0], image.shape[1]))

cv2.fillConvexPoly(mask, pts, 1)
mask = mask.astype(np.bool)

out = np.zeros_like(image)
out[mask] = image[mask]

plt.imshow(out)

##########

screenCnt2 = None
#
## loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
 
    # if our approximated contour has four points, then
    # we can assume that we have found our card
    if len(approx) == 4:
        screenCnt = approx
        continue

a =cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

plt.imshow(a)
plt.show()


pts = screenCnt

mask = np.zeros((image.shape[0], image.shape[1]))

cv2.fillConvexPoly(mask, pts, 1)
mask = mask.astype(np.bool)

out2 = np.zeros_like(image)
out2[mask] = image[mask]

out3 = out- out2
plt.imshow(out3)
plt.show()


