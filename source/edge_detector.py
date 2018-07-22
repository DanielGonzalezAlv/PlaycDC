#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:18:57 2018

@author: PlacDC 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import imutils
import scipy.misc
import os
import glob
from scipy import misc
from skimage.transform import rescale, resize, downscale_local_mean

if __name__=="__main__":

    # Path to use
    #path_cards = "../Data/YOLO/YOLO_img"
    path_textures = "../Data/dtd-r1.0.1/dtd/images/bubbly/"
    i,j=0,0
    for img in glob.glob(os.path.join(path_cards, '*.jpg')):
        i +=1  
        img = misc.imread(img)
        img = rescale(img,1/1)
        img_row = img.reshape(img.shape[0] *img.shape[1] * img.shape[2])
        
        for texture in  glob.glob(os.path.join(path_textures, '*.jpg')):
            j +=1
            texture = misc.imread(texture)
            texture = resize(texture,img.shape)
            

            
            #zero_img = img[:,:,0] + img[:,:,1] + img[:,:,2]  
            #zz = zero_img.reshape(zero_img.shape[0] * zero_img.shape[1])
           #paste = np.zeros(img.shape)
            #i0= img[:,:,0]; i0 = i0.reshape(i0.shape[0]*i0.shape[1])
            #i1= img[:,:,1]; i1 = i1.reshape(i1.shape[0]*i1.shape[1])
            #i2= img[:,:,2]; i2 = i2.reshape(i2.shape[0]*i2.shape[1])

            #t0= texture[:,:,0]; t0 = t0.reshape(t0.shape[0]*t0.shape[1])
            #t1= texture[:,:,1]; t1 = t1.reshape(t1.shape[0]*t1.shape[1])
            #t2= texture[:,:,2]; t2 = t2.reshape(t2.shape[0]*t2.shape[1])

            #p0 = np.add(i0,t0, where= (zz == 0)).reshape(img[:,:,0].shape)
            #p1 = np.add(i1,t1, where= (zz == 0)).reshape(img[:,:,0].shape)
            #p2 = np.add(i2,t2, where= (zz == 0)).reshape(img[:,:,0].shape)
            
            #paste[:,:,0] = np.add(img[:,:,0], texture[:,:,0], where=(zero_img==0))
            #paste[:,:,1] = np.add(img[:,:,1], texture[:,:,1], where=(zero_img==0))
            #paste[:,:,2] = np.add(img[:,:,2], texture[:,:,2], where=(zero_img==0))
            #paste = paste + img
            
            #paste[:,:,0] = p0 
            #paste[:,:,1] = p1 
            #paste[:,:,2] = p2 
            #paste = paste + img
            #PLOT = True 
            #if PLOT: 
            #    plt.imshow(paste)
            #    plt.show()
            


            # CREATE THRESHOLD
            threshold = 0.05
            t2 = img -(np.zeros(img.shape) + threshold)
            t2 = np.maximum(t2, np.zeros(img.shape))
            #plt.imshow(t2); plt.show() 
            t = np.maximum(texture,t2 * 1000) - 999* t2
           # 
           # 
           # 
            file_name_img ="./tmp/"+ str(i) + str(j) + ".jpg"
            scipy.misc.imsave(file_name_img, t)
           
        if i == 7:
            break 
