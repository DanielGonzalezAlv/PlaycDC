#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:31:53 2018


@author: PlaycDC
"""
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc



if __name__ == "__main__":
    canvas_size = (1100, 1100)

    path_files = "../Data/npConvex"
    path_images = "../Data/Images/Post-Processed/"

    img = misc.imread(path_images + np_file[2] + ".jpg")
    
    for filename in glob.glob(os.path.join(path_files, '*.npy')):
        np_file = np.load(filename)
        img = misc.imread(path_images + np_file[2] + ".jpg")

        til = Image.new("RGB", canvas_size, color = 'white')
        # images are of format 600x900, so in order to facilitate nice rotations and so forth we need 1081 pixels canvas
        card_name = np_file[2]
        im = Image.open('../Data/Images/Post-Processed/' + card_name + '.jpg') #25x25
    
        # define an offset at which the image gets pasted.
        midpoint = (im.size[0] / 2 , im.size[1] / 2)
        x_pos = int(canvas_size[0]/2 - midpoint[0])
        y_pos = int(canvas_size[1]/2 - midpoint[1])
    
        til.paste(im, (x_pos, y_pos))
   
        til.save('../Data/Images/pics_on_canvas/' + card_name.lower() + '.jpg')

        np_file[0][:,0] += x_pos
        np_file[0][:,1] += y_pos
        np_file[1][:,0] += x_pos
        np_file[1][:,1] += y_pos
        
        file_save = "../Data/npConvex/" + card_name + "_canvas"
        np.save(file_save, np_file)
       
    #    
#    plt.imshow(img)    
#    plt.scatter(np_file[0][:,0],np_file[0][:,1])

    
