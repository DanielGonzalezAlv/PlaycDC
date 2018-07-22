#!usr/bin/env python3
#-*- coding: utf-8 -*-
# @ PlaycDC

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

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

def plotplot(img, points_hull1, points_hull2):
    plt.imshow(img)
    plt.scatter(points_hull1[:,0], points_hull1[:,1])
    plt.scatter(points_hull2[:,0], points_hull2[:,1])

    plt.show()
    
if __name__ == "__main__":
    
    ###########################
    # Insert your Path and image
    
    path_img = "../Data/Images/Post-Processed/"
<<<<<<< HEAD
    cardname = "qs_2"
=======
    cardname = "2d"
>>>>>>> a31ece3a89a2d4c284d0ad0d42471cbd5886b2ec
    ###########################
    
    img = cv2.imread(path_img + cardname +".JPG")
    
    try:
        if img == None:
            img = cv2.imread(path_img + cardname +".jpg")
    except:
        pass

    height, width, _ = img.shape
            
    # For the first bounding box
<<<<<<< HEAD
    xmin1, xmax1 = 2, 73  
    ymin1, ymax1 = 2, 230
    img1 = img[ymin1:ymax1, xmin1:xmax1, :]
    
    # For the second bounding box
    xmin2, xmax2 = -79, -1     # This values have to be negative
    ymin2, ymax2 = -250, -1     # This values have to be negative
    img2 = img[ymin2:ymax2, xmin2:, :]
=======
    xmin1, xmax1 = 10, 76   
    ymin1, ymax1 = 4, 240
    img1 = img[ymin1:ymax1, xmin1:xmax1, :]
    
    # For the second bounding box
    xmin2, xmax2 = -90, -10     # This values have to be negative
    ymin2, ymax2 = -233, -10     # This values have to be negative
    img2 = img[ymin2:ymax2, xmin2:xmax2, :]
>>>>>>> a31ece3a89a2d4c284d0ad0d42471cbd5886b2ec
    
    # compute convex Hulls 
    hull1, b1 = convex_hull_for_part_of_image(img1)
    hull2, b2 = convex_hull_for_part_of_image(img2)
    
    # Plot
    plot_all_hulls(img, [[hull1, b1], [hull2, b2]], [[xmin1,ymin1],[width+xmin2, height+ymin2]])
   
    # Compute the points of the convex hulls 
    points_hull1 = []
    for vert in hull1.vertices:
        points_hull1.append([b1[vert, 0] + xmin1, b1[vert, 1] + ymin1])
    points_hull1 = np.array(points_hull1)

    points_hull2 = []
    for vert in hull2.vertices:
        points_hull2.append([b2[vert, 0] + width+xmin2, b2[vert, 1] + height + ymin2])
    points_hull2 = np.array(points_hull2)
    
    # Verify points of convex hull
    plotplot(img, points_hull1, points_hull2)
    
    mat_card = [points_hull1, points_hull2, cardname.lower()]

    # save numpy_array 
    file_save = "../Data/npConvex/" + cardname 
    np.save(file_save, mat_card)
