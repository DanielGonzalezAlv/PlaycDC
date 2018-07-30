#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code generates new images using linear transformations.
After this, it crops the images to the middle to reduce their resolution.

Please read README for USAGE information.

@author: Frank Gabel & Daniel Gonzalez
@PlayCDC
"""

#import external dependencies
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.misc
import os
import glob
from skimage.transform import rescale, resize, downscale_local_mean
import sys

def transformation_seq():
    """
    This function defines a sequence of linear transformations to generate
    new data.
    """
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-90, 90),
            )),
            iaa.Affine(
                 scale= (0.75,1.1)
            ),
    ])
    return seq


def get_coords(points, dim_img):
    """
    This function detects the middle point of the convex hulls and
    returns it together with maximum height and width with respect 
    to the convex hull.
    :param points: Points of the convex hull 
    :type points: numpy.ndarrray
    :param dim_img: name of the card
    :type dim_img: numpy.ndarrray
    """
    if points ==[]:
        return [] 
    xmin = points[:,0].min()  
    xmax = points[:,0].max() 
    ymin = points[:,1].min()  
    ymax = points[:,1].max()  

    xmid = ((xmin + xmax) / 2) / dim_img[1]  
    ymid = ((ymin + ymax) / 2) / dim_img[0] 
    width = (xmax - xmin)      / dim_img[1]  
    height = (ymax - ymin)     / dim_img[0]  

    return [xmid, ymid, width, height] 

def create_txt_files(filepath, file_name, list_files1, list_files2):
    """
    This function creates a file with the necessary information for YOLO.
    It writes the labels information together with the Bounding Box positions. 
    """
    f = open(filepath, "w+")

    # First line 
    if list_files1 != []:
        f.write(str(file_name) + " " )
        f.write(str(list_files1[0]) +" " + str(list_files1[1]) + " " + str(list_files1[2]) + " " + str(list_files1[3]) + "\n")

    # Second line 
    if list_files2 != []:
        f.write(str(file_name) + " " )
        f.write(str(list_files2[0]) +" " + str(list_files2[1]) + " " + str(list_files2[2]) + " " + str(list_files2[3]))
    
    f.close()
     
def create_file_info(filename, cards_dict):
    """
    This function writes a .txt files with necessery information for YOLO.
    It needs only to run once, after this, the function can be 
    ommited/commented in the script.
    
    :param filename: name of the file 
    :type filename: string
    :param cards_dict: dictionary of the cards and labels 
    :type carname: dict 
    """
    # Create file mit keys
    f= open(file_name, "w+")
    for names in cards_dict:
        f.write(names) 
        f.write("\n")
    f.close()
    
def cut_convex(conv_hull, range_cut, dim_img_cut):
    """
    This function crops the image into the middle.
    It accepts only convex hulls with enough amount of points that are not
    located to near at the boundary.
    """

    cut_hull = conv_hull - range_cut
    cut_hull = np.maximum(cut_hull, np.zeros(cut_hull.shape))
    cut_hull = np.minimum(cut_hull, dim_img_cut[1] * np.ones(cut_hull.shape))

    #threshold
    threshold = 300
    if np.sum(cut_hull[:,0]) <= (0+threshold) or np.sum(cut_hull[:,1]) <= (0+threshold): 
        cut_hull = []
        return cut_hull 

    max_value = len(cut_hull) * dim_img_cut[1] 
    if np.sum(cut_hull[:,0]) >= (max_value - threshold) or np.sum(cut_hull[:,1]) >= (max_value - threshold):
        cut_hull = []
        return cut_hull

    return cut_hull

if __name__ == "__main__":
      
    if len(sys.argv) == 1:
        print("Generating new data using 5 transormations per image, without ploting the results ...")
        print("Note: if you want to change this values, use parsing.")
        print("For more information check the README file")
        
        # Plot results
        PLOT_results = False

        # transformations
        transformations = 5   

    elif len(sys.argv) == 2:
        transformations = int(sys.argv[1])
        print("Creating data using", transformations, " per image, without plotting the results...")

        # Plot results
        PLOT_results = False

    elif len(sys.argv) == 3:
        transformations = int(sys.argv[1])
        print("Creating data using", transformations, " per image and plotting the results...")

        # Plot results
        PLOT_results = True 

    else:
        print('Parameters are provided in a wrong way.')
        print('For more information, pleas consut the README')
        print('This program ends here!')
        sys.exit()

     
    # Diretory of all the cards
    cards_dict = {"ad":0,"2d":1,"3d":2,"4d":3,"5d":4,"6d":5,"7d":6,"8d":7,"9d":8,"10d":9,"jd":10,"qd":11,"kd":12,
                  "ah":13,"2h":14,"3h":15,"4h":16,"5h":17,"6h":18,"7h":19,"8h":20,"9h":21,"10h":22,"jh":23,"qh":24,"kh":25,
                  "ac":26,"2c":27,"3c":28,"4c":29,"5c":30,"6c":31,"7c":32,"8c":33,"9c":34,"10c":35,"jc":36,"qc":37,"kc":38,
                  "as":39,"2s":40,"3s":41,"4s":42,"5s":43,"6s":44,"7s":45,"8s":46,"9s":47,"10s":48,"js":49,"qs":50,"ks":51}

    # Create file with info of the cards (necessary for YOLO)
    file_name = "../data/" + "info_files.txt" 
    
    # This part of the code was intended to generate a file necessary for YOLO.
    # As this file is already created, this line is commented.
    #create_file_info(file_name, cards_dict)
    
    
    # Paths to work with 
    path_images = "../data/textures/images/"
    path_np_files = "../data/textures/np_convex"
    
    tmp_k = 0
    # decide how much to crop the image
    range_cut = 1100
    for filename in glob.glob(os.path.join(path_np_files, '*.npy')):
        np_file = np.load(filename)
        file_name = filename.replace(path_np_files+"/","").replace(".npy","")
        #print(file_name)
        img = misc.imread(path_images + file_name + ".jpg")

        keypoints1 = ia.KeypointsOnImage([ia.Keypoint( x = np_file[0][i][0], y = np_file[0][i][1]) 
            for i in range(len(np_file[0]))], shape= img.shape)     
        keypoints2 = ia.KeypointsOnImage([ia.Keypoint( x = np_file[1][i][0], y = np_file[1][i][1]) 
            for i in range(len(np_file[1]))], shape= img.shape)     

        # Generate tranformations on cards
        for j in range(transformations):
            seq = transformation_seq()
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
            

            img_aug_cut = img_aug[range_cut:img_aug.shape[0]-range_cut, range_cut:img_aug.shape[1]-range_cut] 
            dim_img_cut = img_aug_cut.shape

            conv_hull1_cut = cut_convex(conv_hull1, range_cut, dim_img_cut)             
            conv_hull2_cut = cut_convex(conv_hull2, range_cut, dim_img_cut)             

            # Plot cards with hulls
            if PLOT_results:
                fig = plt.figure()
                plt.imshow(img_aug_cut); 

                if conv_hull1_cut != []:
                    ## Draw convex hulls
                    xmin = conv_hull1_cut[:,0].min() # Correct? 
                    xmax = conv_hull1_cut[:,0].max() # Correct? 
                    ymin = conv_hull1_cut[:,1].min() # Correct? 
                    ymax = conv_hull1_cut[:,1].max() # Correct? 

                    # PLOT
                    #plt.scatter(conv_hull1_cut[:,0], conv_hull1_cut[:,1]) 
                    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')
                    plt.plot(conv_hull1_cut[:,0], conv_hull1_cut[:,1], color='green')
                    plt.scatter(conv_hull1_cut[:,0], conv_hull1_cut[:,1], color='b', s=10)

                if conv_hull2_cut != []:
                    ## Draw convex hulls
                    xmin = conv_hull2_cut[:,0].min() # Correct? 
                    xmax = conv_hull2_cut[:,0].max() # Correct? 
                    ymin = conv_hull2_cut[:,1].min() # Correct? 
                    ymax = conv_hull2_cut[:,1].max() # Correct? 

                    # PLOT
                    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color='r')
                    plt.plot(conv_hull2_cut[:,0], conv_hull2_cut[:,1], color='green')
                    plt.scatter(conv_hull2_cut[:,0], conv_hull2_cut[:,1], color='b', s=10)
                    #plt.scatter(conv_hull2_cut[:,0], conv_hull2_cut[:,1]) 
                
                #plt.savefig('./tmp/{}.jpg'.format(tmp_k)) 
                #tmp_k += 1
                plt.show()


            # Get data for YOLO
            list_coords1 = get_coords(conv_hull1_cut, img_aug_cut.shape)
            list_coords2 = get_coords(conv_hull2_cut, img_aug_cut.shape)

            # Create .txt files
            # Save first convexhull if it exists      
            # Write on text files
            if np_file[2][-1] == "2":
                card_name = np_file[2][:-2]
                card_name = cards_dict[card_name]
            else:
                card_name = cards_dict[np_file[2]]

            # save textfile
            if list_coords1 == [] and list_coords2 == []:
                print("chech")
                continue
            else:
                file_name_txt = "../YOLO/cards_data/labels/" + file_name + "-" + str(j) + ".txt"
                create_txt_files(file_name_txt, card_name, list_coords1, list_coords2)   
 
                # Save images
                file_name_img = "../YOLO/cards_data/JPEGImages/" + file_name + "-" + str(j)+ ".jpg"
                scipy.misc.imsave(file_name_img, img_aug_cut)
