#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is used to indentify semi automatic the convex hulls of the
suits and ranks of each card

@author: Frank Gabel & Daniel Gonzalez
@PlayCDC
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def detect_convexhull(img, cardname, indx_hull, save=False):
    """
    This function detects one convex hull of the suit and rank of a card
    :param image: card
    :type image: numpy.ndarrray
    :param carname: name of the card
    :type carname: str
    :param indx_hull: number of convex hull of the card
    :type indx_hull: int
    :rparam hull : information of the found convex hull (vertices, area, ...)
    :rtype hull: scipy.spatial.qhull.ConvexHull
    :rparam b: index of the convexhulls on the image
    :rtype b: numpy.ndarray
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray * -1
    img_gray += 255
    ret, thresh = cv2.threshold(img_gray, 150, 1, 0)
    a = np.nonzero(thresh == 1)
    b = np.c_[a[1], a[0]]

    # compute the convex hull
    hull = ConvexHull(b)

    plt.imshow(img)
    for simplex in hull.simplices:
        plt.plot(b[simplex, 0], b[simplex, 1], 'k-')
        plt.scatter(b[simplex, 0], b[simplex, 1], c='r')
        plt.axis('off')
    if save:
        filename = '../data/images/post-processed/samples/\
        {}_{}.jpg'.format(cardname,indx_hull)
        plt.savefig(filename)    
    plt.show()
    return hull, b

def plot_hulls(img, points_hull1, points_hull2, cardname, save=False):
    """
    This function plots the convex hulls of the cards
    :param img: card
    :type img: numpy.ndarray
    :param points_hull1: points of the first convex hull
    :type points_hull1: numpy.ndarray
    :param points_hull2: points of the second convex hull
    :type points_hull2: numpy.ndarray
    :param cardname: name of the card
    :type carname: str
    """
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.plot(np.append(points_hull1[:, 0], points_hull1[:, 0][0]),
             np.append(points_hull1[:, 1], points_hull1[:, 1][0]), 'k')
    plt.scatter(points_hull1[:, 0], points_hull1[:, 1], s=25, c='r',
                edgecolors='k')

    plt.plot(np.append(points_hull2[:, 0], points_hull2[:, 0][0]),
             np.append(points_hull2[:, 1], points_hull2[:, 1][0]), 'k')
    plt.scatter(points_hull2[:, 0], points_hull2[:, 1], s=25, c='r',
                edgecolors='k')

    if save:
        filename = '../data/images/post-processed/samples/{}.jpg'.format(cardname)
        plt.savefig(filename)
    plt.show()

if __name__ == "__main__":

    ###########################
    # Insert your Path and image
    path_img = "../data/images/post-processed/"
    cardname = "qs_2"
    ###########################

    img = cv2.imread(path_img + cardname + ".JPG")
    try:
        if img is None:
            img = cv2.imread(path_img + cardname + ".jpg")
    except:
        pass

    height, width, _ = img.shape

    # Points for finding the first bounding box
    xmin1, xmax1 = 2, 73
    ymin1, ymax1 = 2, 230
    img1 = img[ymin1:ymax1, xmin1:xmax1, :]

    # Points for finding second bounding box
    xmin2, xmax2 = -79, -1     # This values have to be negative
    ymin2, ymax2 = -250, -1     # This values have to be negative
    img2 = img[ymin2:ymax2, xmin2:, :]

    # Compute convex hulls
    indx_hull = 1  # If dectected hull wants to be saved
    hull1, b1 = detect_convexhull(img1, cardname, indx_hull, True)
    indx_hull = 2  # If dectected hull wants to be saved
    hull2, b2 = detect_convexhull(img2, cardname, indx_hull, True)

    points_hull1 = []
    for vert in hull1.vertices:
        points_hull1.append([b1[vert, 0] + xmin1, b1[vert, 1] + ymin1])
    points_hull1 = np.array(points_hull1)

    points_hull2 = []
    for vert in hull2.vertices:
        points_hull2.append([b2[vert, 0] + width+xmin2, b2[vert, 1] +
                             height + ymin2])
    points_hull2 = np.array(points_hull2)

    # Verify points of convex hull
    plot_hulls(img, points_hull1, points_hull2, cardname, True)

    mat_card = [points_hull1, points_hull2, cardname.lower()]

    # save numpy_array
    file_save = "../data/npConvex/" + cardname
    np.save(file_save, mat_card)
