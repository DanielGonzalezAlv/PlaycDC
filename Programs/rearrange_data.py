#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:31:53 2018

@author: PlaycDC
"""
import numpy as np
import os
import glob


if __name__ == "__main__":
    
    path_files = "../Data/npConvex"
  
    np_math = np.zeros([3]) # contains convexhull1 convehull2 label for each row
    
    for filename in glob.glob(os.path.join(path_files, '*.npy')):
        np_file = np.load(filename)
        if len(np_file[2]) > 2:
            np_file[2]= np_file[2][0:2]
        np_math = np.vstack([np_math, np_file])
        

 #    path_files = "../Data/npConvex"
##    file = path_files + "2c_2.npy"
##    ans = np.load(file)