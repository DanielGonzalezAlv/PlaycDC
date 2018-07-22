#!usr/bin/env python3
#-*- coding: utf-8 -*-
# @ PlaycDC

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import os
import glob

if __name__ == "__main__":
    ia.seed(1)
    images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8   
    )
   
    # Import images
    path_np_files = "../Data/npConvex" 
    path_images = "../Data/Images/Post-Processed"

    #misc.imread(path_images +)
    for filename in glob.glob(os.path.join(path_np_files, '*.npy')):
        np_file = np.load(filename)
        print(np_file[2])
         
        #misc.imread(path_images +)
        break 
        
        #if len(np_file[2]) > 2:
        #    np_file[2]= np_file[2][0:2]
        #np_math = np.vstack([np_math, np_file])
