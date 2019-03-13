import cv2
import math
import numpy as np
import scipy.ndimage
import glob
from skimage.io import imread_collection
import os
import errno
import shutil


def main_algorithm(gray_image):
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

 
# source = os.listdir("E:\\CrackTestImg\\")
# destination = "E:\\CrackOp\\"
# for files in source:
#     if files.endswith(".jpeg"):
#         shutil.copy(files,destination)


mydir = 'CrackOp/'
for fil in glob.glob("CrackTestImg/*.jpg"):
    image = cv2.imread(fil) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = main_algorithm(gray_image)
    cv2.imwrite(os.path.join(mydir,fil),result) 