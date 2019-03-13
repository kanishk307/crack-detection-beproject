import cv2
import math
import numpy as np
import scipy.ndimage
# from skimage import io, morphology, img_as_bool, segmentation
# # from scipy import ndimage as ndi
# import matplotlib.pyplot as plt

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
count_white = 0
input_img = cv2.imread('a3.jpg')
gray_image = cv2.imread(r'a3.jpg', 0)


with_nmsup = True #apply non-maximal suppression
fudgefactor = 1 #with this threshold you can play a little bit
sigma = 21 #for Gaussian Kernel
kernel = 2*math.ceil(2*sigma)+1 #Kernel size

gray_image = gray_image/255.0
blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
gray_image = cv2.subtract(gray_image, blur)

# compute sobel response
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=31)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=31)
mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)

# threshold
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0

#either get edges directly
if with_nmsup is False:
    mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('im', result)
    cv2.imshow('Input',input_img)
    cv2.waitKey()

#or apply a non-maximal suppression
else:
    # non-maximal suppression
    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)


    # imgray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(result,127,255,0)
    # _, contours, _= cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # _, contours, _= cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # resultContour=cv2.drawContours(result,contours,-1,(0,255,0),3)
    # print("no of shapes {0}".format(len(contours)))

    # for cnt in contours:
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     input_img = np.int0(box)
    #     res = cv2.drawContours(result, [box.astype(int)],0,(0,255,0),3)






    #result = cv2.erode(result, result, kernel)
    cv2.imshow('res', result)
    cv2.imshow('cont', res)
    # cv2.imshow('resultWContours', resultContour)
    # cv2.imshow('out', output)
    # cv2.imshow('im', e_im)
    count_black = np.sum(result==0)
    count_white = np.sum(result == 255)
    print('Number of white pixels : ', count_white)
    print('Number of black pixels : ', count_black)
    total_pixels = count_black + count_white
    crack_ratio = count_white / total_pixels
    crack_intensity = crack_ratio * 100
    print('Crack Intensity is :', crack_intensity)
    cv2.imshow('Input',input_img)
    cv2.waitKey()
