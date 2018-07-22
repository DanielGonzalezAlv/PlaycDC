#!usr/bin/env python3
#-*- coding: utf-8 -*-
# @ PlaycDC

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
            # Work with the color
            #sometimes(iaa.WithColorspace(to_colorspace="HSV", from_colorspace= "RGB",
            #                   children=iaa.WithChannels(1, iaa.Add(100)))),
            
            # crop some of the images by 0-10% of their height/width
            #sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
           # sometimes(iaa.Affine(
            #    scale={"x": (0.5, 0.7), "y": (0.5, 0.7)},
#                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            #    rotate=(-90, 90),
            #    #shear=(-16, 16),
            #    order=[0, 1],
            #    #cval=(0, 255),
           # )),

            # Sharpen each image, overlay the result with the original
            # image using an alpha between 0 (no sharpening) and 1
            # (full sharpening effect).
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.25, 3.5)),
#
#            # Same as sharpen, but for an embossing effect.
#            #iaa.Emboss(alpha=(0, 1.0), strength=(0, 5.0)),
#
#            # Blur each image with varying strength using
#            # gaussian blur (sigma between 0 and 3.0),
#            # average/uniform blur (kernel size between 2x2 and 7x7)
#            # median blur (kernel size between 3x3 and 11x11).
            iaa.OneOf([
                iaa.GaussianBlur((0, 6.0)),
                iaa.AverageBlur(k=(0, 5)),
                iaa.MedianBlur(k=(1, 11)),
                ]),
#            
            # Search in some images either for all edges or for
            # directed edges. These edges are then marked in a black
            # and white image and overlayed with the original image
            # using an alpha of 0 to 0.7.
            #sometimes(iaa.OneOf([
            #    iaa.EdgeDetect(alpha=(0, 0.7)),
            #    iaa.DirectedEdgeDetect(
            #    alpha=(0, 0.7), direction=(0.0, 1.0)
            #    ),
            #    ])),

            # Add gaussian noise to some images.
            # In 50% of these cases, the noise is randomly sampled per
            # channel and pixel.
            # In the other 50% of all cases it is sampled once per
            # pixel (i.e. brightness change).
            #iaa.AdditiveGaussianNoise(
            #    loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

             #Invert each image's chanell with 5% probability.
             #This sets each pixel value v to 255-v.
             #  iaa.Invert(0.05, per_channel=True), # invert color channels

             # Add a value of -10 to 10 to each pixel.
             #  iaa.Add((-10, 10), per_channel=0.5),

             # Change brightness of images (50-150% of original value).
             #iaa.Multiply((0.5, 1.5), per_channel=0.5),

             # Improve or worsen the contrast of images.
             #  iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                  #  iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                  #  sometimes(
                  #      iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                  #  ),

                # In some images distort local areas with varying strength.
                  #  sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
         #       ],
            # do all of the above augmentations in random order
 #               random_order=True
 #           )
 #       ],
    # do all of the above augmentations in random order
 #       random_order=True
    ])
    return seq

def seq_rotation():
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.1, 0.6), "y": (0.1, 0.6)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-90, 90),
            #    #shear=(-16, 16),
            #    order=[0, 1],
            #    #cval=(0, 255),
            )),

    ])
    return seq

if __name__ == "__main__":

    
#    Paths to work
#    path_np_files = "../Data/npConvex_on_canvas" 
#    path_images = "../Data/Images/pics_on_canvas/"  # pics on cavas
    path_np_files = "../Data/npConvex" 
    path_images = "../Data/Images/Post-Processed/" 
    path_textures = "../Data/dtd-r1.0.1/my_textures_zoomedout"

    images_per_texture = 3 
    for subdir, dirs, files in os.walk(path_textures):
        for directories in dirs:
            name_dir_textures = path_textures + "/" + directories + "/"
            for filename in glob.glob(os.path.join(path_np_files, '*.npy')):
                np_file = np.load(filename)
                img = misc.imread(path_images + np_file[2] + ".jpg")
                
                # shift value 

                texture_canvas = [3000,3000]
                shift = [int((texture_canvas[0]-img.shape[0])/2), int( (texture_canvas[1]-img.shape[1])/2)]

                np_file[0][:,0] = np_file[0][:,0] + shift[1]
                np_file[0][:,1] = np_file[0][:,1] + shift[0]

                np_file[1][:,0] = np_file[1][:,0] + shift[1]
                np_file[1][:,1] = np_file[1][:,1] + shift[0]

                # Get texture 
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
                         
                    # Save images
                    file_name_img = "../Data/Textures/Images_zoomedout/" + str(np_file[2]) + "-"+ name_of_texture + ".jpg"
                    scipy.misc.imsave(file_name_img, image_aug)

                    # save npfiles
                    file_name_np = "../Data/Textures/np_convex_zoomedout/" + str(np_file[2]) + "-"+ name_of_texture 
                    np.save(file_name_np, np_file)

                    #break for textures
                    break_texture +=1
                    if break_texture == images_per_texture:
                        break
