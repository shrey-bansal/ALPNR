import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.measure import regionprops
from skimage import measure
from skimage.util import img_as_ubyte


# Image operation using thresholding
def Plate_Letter_List(License_Plate_Gray):
    Letter_List = []
    License_Plate = License_Plate_Gray < 150



    ret, thresh = cv2.threshold(License_Plate, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                kernel, iterations = 2)

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations = 1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    cv2.imshow('image', fg)
