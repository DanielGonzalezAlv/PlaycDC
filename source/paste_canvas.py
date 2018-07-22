#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program perform blurring, sharping and change of lightness to the 
cards and later paste it into canvanses provided by DDT
(see: https://github.com/datadriventests/ddt).
For this is important to save the dtd images in the "../data/" directory

@author: Frank Gabel & Daniel Gonzalez
@PlayCDC
"""

# import external dependencies
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.misc
import os
import glob
from skimage.transform import rescale, resize, downscale_local_mean

def my_seq():
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential(
        [
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.25, 3.5)),
            iaa.OneOf([
                iaa.GaussianBlur((0, 6.0)),
                iaa.AverageBlur(k=(0, 5)),
                iaa.MedianBlur(k=(1, 11)),
                ]),
    ])
    return seq

if __name__ == "__main__":

    
    # Paths to work
    path_np_files = "../data/npConvex" 
    path_images = "../data/images/post-processed/" 
    path_textures = "../data/dtd-r1.0.1/dtd/images"  # Make sure to download the data first

    # images per directory
    images_per_texture = 1 

    # number of directories to go through
    ndirs = 5

    for subdir, dirs, files in os.walk(path_textures):
        break_directory = 0
        for directories in dirs:
            name_dir_textures = path_textures + "/" + directories + "/"

            # Take in care with how many directories to work with
            break_directory += 1
            if break_directory == ndirs:
                break
            
            for filename in glob.glob(os.path.join(path_np_files, '*.npy')):
                np_file = np.load(filename)
                img = misc.imread(path_images + np_file[2] + ".jpg")
                
                # shift the values of the convex hulls 
                texture_canvas = [3000,3000]
                shift = [int((texture_canvas[0]-img.shape[0])/2), int( (texture_canvas[1]-img.shape[1])/2)]

                np_file[0][:,0] = np_file[0][:,0] + shift[1]
                np_file[0][:,1] = np_file[0][:,1] + shift[0]

                np_file[1][:,0] = np_file[1][:,0] + shift[1]
                np_file[1][:,1] = np_file[1][:,1] + shift[0]

                # Use texture 
                break_texture = 0 
                for texture in  glob.glob(os.path.join(name_dir_textures, '*.jpg')):
                    # Texture
                    name_of_texture = texture.replace(name_dir_textures ,"").replace(".jpg", "")

                    texture = misc.imread(texture)
                    texture = resize(texture,(texture_canvas[0],texture_canvas[1]))

                    seq = my_seq()
                    seq_det = seq.to_deterministic()
                    img_aug = seq_det.augment_images([img])[0]
                        
                    # Use textures
                    img_aug = img_aug / img_aug.max() 
                    texture[shift[0]:shift[0]+img_aug.shape[0],shift[1]:shift[1]+img_aug.shape[1]] = img_aug
                    image_aug = texture
                         
                    ##### Save
                    # Save images
                    file_name_img = "../data/textures/final_try/" + str(np_file[2]) + "-"+ name_of_texture + ".jpg"
                    scipy.misc.imsave(file_name_img, image_aug)

                    # save npfiles
                    file_name_np = "../data/textures/np_convex_final_try/" + str(np_file[2]) + "-"+ name_of_texture 
                    np.save(file_name_np, np_file)

                    #break for textures
                    break_texture +=1
                    if break_texture == images_per_texture:
                        break