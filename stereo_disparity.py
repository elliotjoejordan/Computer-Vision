import cv2
import os
import numpy as np

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

directory_to_cycle_left = "left-images";
directory_to_cycle_right = "right-images";

# Function to return disparity image for a single file

def calculateDisparity(filename_left, master_path_to_dataset, skip_forward_file_pattern):

    # Create paths to read files

    full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
    full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);


    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # Read in images

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # Convert to greyscale

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # Compute disparity

        disparity = stereoProcessor.compute(grayL,grayR)

        # Filter noise

        dispNoiseFilter = 5
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # Return file's disparity image

        return disparity_scaled
    return None

#cv2.imshow("disparity", calculateDisparity('1506942473.484027_L.png', "../TTBB-durham-02-10-17-sub10", ""))
#cv2.waitKey()