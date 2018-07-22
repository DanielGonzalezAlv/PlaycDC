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

def my_seq():
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential(
        [
            # Work with the color
            sometimes(iaa.WithColorspace(to_colorspace="HSV", from_colorspace= "RGB",
                               children=iaa.WithChannels(1, iaa.Add(100)))),
            
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
            sometimes(iaa.Affine(
                scale={"x": (0.5, 0.7), "y": (0.5, 0.7)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-90, 90),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
            )),

            # Sharpen each image, overlay the result with the original
            # image using an alpha between 0 (no sharpening) and 1
            # (full sharpening effect).
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.25, 5.5)),

            # Same as sharpen, but for an embossing effect.
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 5.0)),

            # Blur each image with varying strength using
            # gaussian blur (sigma between 0 and 3.0),
            # average/uniform blur (kernel size between 2x2 and 7x7)
            # median blur (kernel size between 3x3 and 11x11).
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
                ]),
            
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

def get_coords(points, dim_img):
    """
    This function take a bunch of points and returns:
    - "mid_point" : Midpoint given with the help of the max min coords of x and y 
    - width 
    - height
    """
    xmin = points[:,0].min() # Correct? 
    xmax = points[:,0].max() # Correct? 
    ymin = points[:,1].min() # Correct? 
    ymax = points[:,1].max() # Correct? 

    if xmax > dim_img[1]: 
        xmax = dim_img[1]
    if xmin < 0: 
        xmin = 0
    if ymax > dim_img[0]: 
        ymax = dim_img[0]
    if ymin < 0: 
        ymin = 0
    xmid = ((xmin + xmax) / 2) / dim_img[1]  
    ymid = ((ymin + ymax) / 2) / dim_img[0] 
    width = (xmax - xmin) / dim_img[1]  
    height = (ymax - ymin)/ dim_img[0]  

    return [xmid, ymid, width, height] 

def create_txt_files(filepath, file_name, list_files1, list_files2):
    """
    This Program writes .txt files for YOLO 

    """
    f = open(filepath, "w+")

    # First line 
    f.write(str(file_name) + " " )
    f.write(str(list_files1[0]) +" " + str(list_files1[1]) + " " + str(list_files1[2]) + " " + str(list_files1[3]) + "\n")

    # Second line 
    f.write(str(file_name) + " " )
    f.write(str(list_files2[0]) +" " + str(list_files2[1]) + " " + str(list_files2[2]) + " " + str(list_files2[3]))
    
    f.close()
     
def create_file_info(filename, cards_dict):
    """
    This function creates a file with the info of the cards
    """
    # Create file mit keys
    f= open(file_name, "w+")
    for names in cards_dict:
        f.write(names) 
        f.write("\n")
    f.close()
    

if __name__ == "__main__":

    cards_dict = {"ad":0,"2d":1,"3d":2,"4d":3,"5d":4,"6d":5,"7d":6,"8d":7,"9d":8,"10d":9,"jd":10,"qd":11,"kd":12,
                  "ah":13,"2h":14,"3h":15,"4h":16,"5h":17,"6h":18,"7h":19,"8h":20,"9h":21,"10h":22,"jh":23,"qh":24,"kh":25,
                  "ac":26,"2c":27,"3c":28,"4c":29,"5c":30,"6c":31,"7c":32,"8c":33,"9c":34,"10c":35,"jc":36,"qc":37,"kc":38,
                  "as":39,"2s":40,"3s":41,"4s":42,"5s":43,"6s":44,"7s":45,"8s":46,"9s":47,"10s":48,"js":49,"qs":50,"ks":51}

    # Create File with info of the cards 
    file_name = "../Data/YOLO/" + "info_files.txt" 
    create_file_info(file_name, cards_dict)
    
#    Paths to work
    path_np_files = "../Data/npConvex_on_canvas" 
    path_images = "../Data/Images/pics_on_canvas/"  # pics on cavas
    #path_np_files = "../Data/npConvex" 
    #path_images = "../Data/Images/Post-Processed/"  # pics on cavas

    for filename in glob.glob(os.path.join(path_np_files, '*.npy')):
        np_file = np.load(filename)
        img = misc.imread(path_images + np_file[2] + ".jpg")
        #img_many = np.vstack((img,img) for _ in range(16))
        
        keypoints1 = ia.KeypointsOnImage([ia.Keypoint( x = np_file[0][i][0], y = np_file[0][i][1]) 
            for i in range(len(np_file[0]))], shape= img.shape)     
        keypoints2 = ia.KeypointsOnImage([ia.Keypoint( x = np_file[1][i][0], y = np_file[1][i][1]) 
            for i in range(len(np_file[1]))], shape= img.shape)     

        for j in range(5):
            seq = my_seq()
            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images([img])[0]
            keypoints1_aug = seq_det.augment_keypoints([keypoints1])[0]
            keypoints2_aug = seq_det.augment_keypoints([keypoints2])[0]
            
            # Get coords as an array
            conv_hull1 = keypoints1_aug.get_coords_array() 
            conv_hull2 = keypoints2_aug.get_coords_array() 

            # Draw keypoints on images for ploting            
            image_before1 = keypoints1.draw_on_image(img, size=10)
            image_before2 = keypoints2.draw_on_image(img, size=10)

            image_after1 = keypoints1_aug.draw_on_image(img_aug,size=10)
            image_after2 = keypoints2_aug.draw_on_image(img_aug,size=10)
            
            # Plot        
            PLOT = False 
            if PLOT:
                plt.subplot(221), plt.imshow(image_before1)
                plt.subplot(222), plt.imshow(image_before2)
                plt.subplot(223), plt.imshow(image_after1)
                plt.subplot(224), plt.imshow(image_after2)
                plt.show()  

            # Get data for YOLO
            list_coords1 = get_coords(conv_hull1, img.shape)
            list_coords2 = get_coords(conv_hull2, img.shape)

            # Create .txt files
            # Save first convexhull if it exists        #TODO CHECK FOR EXISTENCE

            # Write on text files
            if np_file[2][-1] == "2":
                card_name = np_file[2][:-2]
                card_name = cards_dict[card_name]
            else:
                card_name = cards_dict[np_file[2]]
    
            # save textfile
            file_name_txt = "../Data/YOLO/YOLO_txt/" + str(np_file[2]) + "-" + str(j) + ".txt"
            create_txt_files(file_name_txt, card_name, list_coords1, list_coords2)   
 
            # Save images
            file_name_img = "../Data/YOLO/YOLO_img/" + str(np_file[2]) + "-" + str(j)+ ".jpg"
            scipy.misc.imsave(file_name_img, img_aug)
        
         
