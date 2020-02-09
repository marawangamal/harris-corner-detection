
from Harris_Corner_Detector import *
from scipy import ndimage
import numpy as np
import cv2
from PIL import Image

if __name__ == '__main__':

    # Import image
    img = cv2.imread("building.jpg")

    # Returns corners overlaid onto image
    hcd = Harris_Corner_Detector(sigma =3, corner_thresh=0.01, k=0.04)
    img_corners = hcd.disp_corners(img)

    cv2.imwrite("building_kps.jpg", img_corners)
