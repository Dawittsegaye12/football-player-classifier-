import numpy as np
import cv2
import pywt

def w2d (img, mode = 'haar', level = 1):
    
    im_array = img
    
    im_array = cv2.cvtColor(im_array,cv2.COLOR_BGR2GRAY)
    
    im_array = np.float32(im_array)
    
    im_array /= 255
    
    coeffs = pywt.wavedec2(im_array,mode,level)
    
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    
    im_array_H = pywt.waverec2(coeffs_H,mode)
    im_array_H *= 255
    im_array_H = np.uint8(im_array_H)
    
    return im_array_H