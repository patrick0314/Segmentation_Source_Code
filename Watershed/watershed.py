#Import modules
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def WaterShed(in_path, out_path):
    # Reading image using cv2 libraries
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Noise Removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations=5)

    # Background area 
    sure_bg = cv2.dilate(opening, kernel, iterations=7)

    # Foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Applying watershed algorithm and marking the regions segmented
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [0,255,255]

    # Displaying the segmented image
    #imS = cv2.resize(img, (612, 368))
    #cv2.imshow('Segmented Result', imS)
    #cv2.waitKey()

    # Save the segmented image
    cv2.imwrite(out_path, img)

if __name__ == '__main__':
    input_folder = 'pics'
    output_folder = 'results'
    filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    for filename in filenames:
        WaterShed(os.path.join(input_folder, filename), os.path.join(output_folder, filename))