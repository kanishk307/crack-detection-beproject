import cv2
import math
import numpy as np
import scipy.ndimage

def main_function(gray_image):
    def orientated_non_max_suppression(mag, ang):
        ang_quant = np.round(ang / (np.pi/4)) % 4
        winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
        winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        magE = non_max_suppression(mag, winE)
        magSE = non_max_suppression(mag, winSE)
        magS = non_max_suppression(mag, winS)
        magSW = non_max_suppression(mag, winSW)

        mag[ang_quant == 0] = magE[ang_quant == 0]
        mag[ang_quant == 1] = magSE[ang_quant == 1]
        mag[ang_quant == 2] = magS[ang_quant == 2]
        mag[ang_quant == 3] = magSW[ang_quant == 3]
        return mag

    def non_max_suppression(data, win):
        data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
        data_max[data != data_max] = 0
        return data_max

    # start calulcation
    # count_white = 0

    # with_nmsup = True #apply non-maximal suppression
    # fudgefactor = 1 #with this threshold you can play a little bit
    # sigma = 21 #for Gaussian Kernel
    # kernel = 2*math.ceil(2*sigma)+1 #Kernel size

    gray_image = gray_image/255.0
    blur = cv2.GaussianBlur(gray_image, (85, 85), 21)
    gray_image = cv2.subtract(gray_image, blur)

    # compute sobel response
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=31)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    # threshold
    threshold = 4 * 1 * np.mean(mag)
    mag[mag < threshold] = 0

    #either get edges directly
    # mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    # kernel = np.ones((5,5),np.uint8)
    # result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('im', result)
        # cv2.imshow('Input',input_img)
        # cv2.waitKey()
    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    return result
        

    #or apply a non-maximal suppression
    # else:
    #     # non-maximal suppression
    #     mag = orientated_non_max_suppression(mag, ang)
    #     # create mask
    #     mag[mag > 0] = 255
    #     mag = mag.astype(np.uint8)
    #     kernel = np.ones((5,5),np.uint8)
    #     result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
        
        
    #    #TRYING FLOODFILL
    #    th, im_th = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY_INV);
    #    h, w = im_th.shape[:2]
    #    mask = np.zeros((h+2, w+2), np.uint8)
    #    
    #    cv2.floodFill(result, mask, (0,0), 255);
    #    im_floodfill_inv = cv2.bitwise_not(result)
    #    new = cv2.bitwise_and(result,im_floodfill_inv)
    ##    new = im_th+im_floodfill_inv
    ##    im_out = cv2.add(im_th,im_floodfill_inv)
    ##    im_out = im_th | im_floodfill_inv
    #    
    ##    cv2.imshow("Thresholded Image", im_th);
    #    cv2.imshow("Floodfilled Image", result);
    #    cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv);
    #    cv2.imshow("Foreground", new);
    #    # FLOODFILL END
        
        
        #result = cv2.erode(result, result, kernel)
    # cv2.imshow('im', result)
        # count_black = np.sum(result==0)
        # count_white = np.sum(result == 255)
        # print('Number of white pixels : ', count_white)
        # print('Number of black pixels : ', count_black)
        # total_pixels = count_black + count_white
        # crack_ratio = count_white / total_pixels
        # crack_intensity = crack_ratio * 100
        # print('Crack Intensity is :', crack_intensity)
    # cv2.imshow('Input',input_img)
    # cv2.waitKey()

    


# input_img = cv2.imread('6.jpg')
gray_image = cv2.imread(r'6.jpg', 0)

result = main_function(gray_image)

cv2.imshow('op',result)
cv2.waitKey()
