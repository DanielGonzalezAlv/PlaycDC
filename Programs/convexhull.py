#!usr/bin/env python3
#-*- coding: utf-8 -*-
# @ PlaycDC

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

###########################
# Insert your Path and image

path_img = "../Data/Images/Post-Processed/"
cardname = "2c_2"
img = cv2.imread(path_img + cardname +".JPG")
###########################

height, width, _ = img.shape

def convex_hull_for_part_of_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray * -1
    img_gray += 255
    ret, thresh = cv2.threshold(img_gray, 150, 1, 0)
    a = np.nonzero(thresh == 1)
    b = np.c_[a[1], a[0]] # b: tatsächlcihe coords
    
    
    hull = ConvexHull(b)
    
    plt.imshow(img)
    for simplex in hull.simplices:
        plt.plot(b[simplex, 0], b[simplex, 1], 'k-')
    plt.show()
    return hull, b

def plot_all_hulls(img, hulls, offsets):
    plt.imshow(img)
    for (hull, b), offset in zip(hulls, offsets):
        for simplex in hull.simplices:
            plt.plot(b[simplex, 0] + offset[0], b[simplex, 1]+ offset[1], 'k-')        
    plt.show()

# For the first bounding box
xmin_1 = 4     
xmax_1 = 100
ymin_1 = 4
ymax_1 = 230

img1 = img[ymin_1:ymax_1, xmin_1:xmax_1, :]

# For the second bounding box
xmin_2 = -100     
xmax_2 = -1
ymin_2 = -220
ymax_2 = -1
img2 = img[ymin_2:ymax_2, xmin_2:, :]

# compute convex Hulls 
hull1, b1 = convex_hull_for_part_of_image(img1)
hull2, b2 = convex_hull_for_part_of_image(img2)

# Plot
plot_all_hulls(img, [[hull1, b1], [hull2, b2]], [[xmin_1, ymin_1], [width-xmax_1, height-ymax_1]])


# save numpy_array 
#file_save = "../Data/npConvex/" + cardname
#np.save(file_save, b1)