import cv2

import numpy as np

def shift_image(image, choice_X, choice_Y):
    """Shifts image left-right or up-down direction 
        input Param: 
            1) image - image array
            2) choice_X - Value of shifting in X axis (in pixel)
            3) choice_Y - Value of shifting in Y axis (in pixel)"""
    try:
        height, width, c = image.shape
    except:
        height, width= image.shape
    T = np.float32([[1, 0, choice_X], [0, 1, choice_Y]])
    output = cv2.warpAffine(image, T, (width, height))
    
    return output

