#####################################################################

# Example : performs selective search bounding box identification

# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Copyright (c) 2018 Department of Computer Science, Durham University, UK

# License: MIT License

# ackowledgements: based on the code and examples presented at:
# https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

#####################################################################

import cv2
import os
import sys
import math
import numpy as np


#####################################################################

# press all the go-faster buttons - i.e. speed-up using multithreads
def search(filename):
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    # create Selective Search Segmentation Object using default parameters

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    #####################################################################
    if '.png' in filename:

        # read image from file

        frame = cv2.imread(filename, cv2.IMREAD_COLOR)


        # start a timer (to see how long processing and display takes)

        # start_t = cv2.getTickCount();

        # set input image on which we will run segmentation

        ss.setBaseImage(frame)

        # Switch to fast but low recall Selective Search method
        ss.switchToSelectiveSearchFast()

        # Switch to high recall but slow Selective Search method (slower)
        # ss.switchToSelectiveSearchQuality()

        # run selective search segmentation on input image
        rects = ss.process()

        return(rects)


#####################################################################