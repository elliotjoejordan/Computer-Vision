################################################################################

# functionality: perform detection based on HOG feature descriptor / SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Minor portions: based on fork from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2
import os
import numpy as np
import math
import params
from utils import *
from sliding_window import *
import stereo_disparity as sd
from stereo_to_3d import project_disparity_to_3d
from selective_search import search



################################################################################

directory_to_cycle = "../FromTown";

################################################################################

# load SVM from file

try:
    svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
except:
    print("Missing files - SVM!");
    print("-- have you performed training to produce these files ?");
    exit();

# print some checks

print("svm size : ", len(svm.getSupportVectors()))
print("svm var count : ", svm.getVarCount())

################################################################################

# process all images in directory (sorted by filename)

for filename in sorted(os.listdir(directory_to_cycle + '/left-images')):

    # if it is a PNG file
    allDepths = []
    if '.png' in filename:
        print(" ")
        print(os.path.join(directory_to_cycle + '/left-images', filename));

        # read image data

        img = cv2.imread(os.path.join(directory_to_cycle + '/left-images', filename), cv2.IMREAD_COLOR)

        # make a copy for drawing the output

        output_img = img.copy();

        # Get image size variables

        img_height, img_width, _ = img.shape

        # Code to alter brightness and contrast for image pre-processing to improve detection

        brightness = 0
        contrast = 70
        img = np.int16(img)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)



        # Commented code applies a sobel filter to the image

        '''
        current = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        abs_current = np.absolute(current)
        sobel8U = np.uint8(abs_current)
        img = sobel8U
        '''

        # Calls selective search to give a list of rectangles

        rectangles = search(directory_to_cycle + "/left-images/" + filename)

        detections = []

        for rectangle in rectangles:

            # For each rectangle, decide whether it is a pedestrian

            x, y, w, h = rectangle

            # Rule out selections that are wider than they are tall, too tall for their width or too small overall

            if w > 3 * h/4:
                rectangles = np.setdiff1d(rectangles, [rectangle])
            elif w < h/5:
                rectangles = np.setdiff1d(rectangles, [rectangle])
            elif w < 30 or h < 50:
                rectangles = np.setdiff1d(rectangles, [rectangle])

            # Do not process if rectangle lies in top or bottom 70 pixels of the image (sky/bonnet)

            elif y < 70 or y+h > img_height-70:
                rectangles = np.setdiff1d(rectangles, [rectangle])
            else:
                # Create a window of the image the size of the rectangle
                window = img[y:y + h, x:x + w]

                img_data = ImageData(window)
                img_data.compute_hog_descriptor()

                # generate and classify each window by constructing a HOG histogram and passing it through the SVM classifier

                if img_data.hog_descriptor is not None:


                    retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))

                    # if we get a detection, then record it

                    if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                        # store rect as (x1, y1) (x2,y2) pair

                        rect = np.float32([x, y, x + w, y + h])


                        detections.append(rect)

                ########################################################

        # For the overall set of detections (over all scales) perform non maximal suppression (i.e. remove overlapping boxes etc).

        detections = non_max_suppression_fast(np.int32(detections), 0.4)

        # create a disparity image based on the two camera feeds

        disparityIMG = sd.calculateDisparity(filename, directory_to_cycle, "")

        # Get the depth (Z points) for pixels in the disparity image

        points = project_disparity_to_3d(disparityIMG, 128, output_img)

        # Draw all the detection on the original image

        for rect in detections:
            depth = -1  # Set initial depth

            width = rect[2]-rect[0]
            length = rect[3]-rect[1]

            # These parameters define the middle section of the image (half width and height, in the center)
            startX = int(round(rect[0] + width / 4))
            endX = int(round(rect[0] + 3 * width / 4))
            startY = int(round(rect[1] + length / 4))
            endY = int(round(rect[1] + 3 * length / 4))


            centerTotalDepth = 0
            count1 = 0

            # Loop through central pixels to calculate closest depth, and total depth of the area (for average)

            for x in range(startX, endX):
                for y in range(startY, endY):
                    try:
                        # Add depth to tally
                        centerTotalDepth += points[(x, y)]
                        count1 += 1

                        # If depth in new nearest point, replace depth with new point

                        if points[(x, y)] < depth or depth < 0:
                            depth = points[(x, y)]
                    except:
                        pass

            # Creates 3x3 boxes in top corners of the rectangle
            # Adds depths to a total to calculate average depths at the corners

            outsideTotalDepth = 0
            count2 = 0

            for x in range(rect[0]-3, rect[0]):
                for y in range(rect[1], rect[1]+3):
                    try:
                        outsideTotalDepth += points[(x,y)]
                        count2 += 1
                    except:
                        pass

            for x in range(rect[2], rect[2]+3):
                for y in range(rect[1]-3, rect[1]):
                    try:
                        outsideTotalDepth += points[(x,y)]
                        count2 += 1
                    except:
                        pass

            # Calculate the average depths (center, corners)

            try:
                middleAVG = centerTotalDepth/count1
            except:
                middleAVG = -2

            try:
                edgeAVG = outsideTotalDepth/count2
            except:
                edgeAVG = -1

            # Filters boxes - must be within 30m of car and non-zero
            # Must also be closer in the middle than at the edge by 25cm (filters out flat surfaces, trees etc)
            if 0 < depth < 30 and middleAVG < edgeAVG - 0.25:
                # Writes depth and box to output image
                allDepths.append(depth)
                cv2.putText(output_img, str("%.1fm" % round(depth, 1)), (rect[0], rect[3]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

            # Deal with the cases where objects are identified in the left of the image where there are no disparity readings
            elif rect[0] < 50 and middleAVG < edgeAVG - 0.25:
                # Images in this region have no depth, so 1m is printed
                cv2.putText(output_img, "1.0m", (rect[0], rect[3]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        # Reveals output image with boxes
        try:
            closest = min(allDepths)
        except:
            closest = 0.0
        filename_right = filename.replace("_L", "_R")
        print(os.path.join(directory_to_cycle + '/right-images', filename_right) + " : nearest detected scene object " + str("%.1fm" % round(closest, 1)))

        cv2.imshow('detected objects',output_img)
        cv2.imshow("Disparity", disparityIMG)


        key = cv2.waitKey(1)
        if (key == ord('x')):
            break




# close all windows

cv2.destroyAllWindows()

#####################################################################
