from PyQt5 import QtWidgets, QtGui, QtCore
# from PyQt5.QtWidgets import (QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QApplication,QInputDialog, QLineEdit, QFileDialog, QLabel)
# from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import math
import numpy as np
import scipy.ndimage
from random import randint

font_but = QtGui.QFont()
font_but.setFamily("Segoe UI Symbol")
font_but.setPointSize(10)
font_but.setWeight(95)
# imagePath=""


class PushBut1(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super(PushBut1, self).__init__(parent)
        self.setMouseTracking(True)
        self.setStyleSheet("margin: 1px; padding: 200px; background-color: rgba(0,0,0,100); color: rgba(0,190,255,255); border-style: solid;"
                           "border-radius: 3px; border-width: 0.5px; border-color: rgba(127,127,255,255);")

    def enterEvent(self, event):
        if self.isEnabled() is True:
            self.setStyleSheet("margin: 1px; padding: 200px; background-color: rgba(0,0,0,100); color: rgba(0,0,150,255);"
                               "border-style: solid; border-radius: 3px; border-width: 0.5px; border-color: rgba(0,0,255,255);")
        if self.isEnabled() is False:
            self.setStyleSheet("margin: 1px; padding: 200px; background-color: rgba(0,0,0,100); color: rgba(0,0,0,255); border-style: solid;"
                               "border-radius: 3px; border-width: 0.5px; border-color: rgba(127,127,255,255);")

    def leaveEvent(self, event):
        self.setStyleSheet("margin: 1px; padding: 200px; background-color: rgba(0,0,0,100); color: rgba(0,0,0,255); border-style: solid;"
                           "border-radius: 3px; border-width: 0.5px; border-color: rgba(127,127,255,255);")


class PyQtApp(QtWidgets.QWidget):
    imagePath=""
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle("Crack Extraction")
        self.setWindowIcon(QtGui.QIcon("D:\Projects\PythonGUI\ELMLogo.jpg"))
        self.setMinimumWidth(resolution.width() / 3)
        self.setMinimumHeight(resolution.height() / 1.5)
        self.setStyleSheet("QWidget {background-color: rgba(200,200,200,255);} QScrollBar:horizontal {width: 1px; height: 1px;"
                           "background-color: rgba(0,41,59,255);} QScrollBar:vertical {width: 1px; height: 1px;"
                           "background-color: rgba(0,41,59,255);}")
        # self.textf = QtWidgets.QTextEdit(self)
        # self.textf.setPlaceholderText("Results...")
        # self.textf.setStyleSheet("margin: 1px; padding: 7px; background-color: rgba(0,255,255,100); color: rgba(0,190,255,255);"
        #                          "border-style: solid; border-radius: 3px; border-width: 0.5px; border-color: rgba(0,140,255,255);")

        # self.layout = QVBoxLayout()
        # self.label = QLabel("My text")

        # self.layout.addWidget(self.label)
        # self.setWindowTitle("My Own Title")
        # self.setLayout(self.layout)


        self.but1 = PushBut1(self)
        self.but1.setText("Upload and Process")
        # self.but1.setStyleSheet("margin-top: 200px;")
        self.but1.setFixedWidth(600)
        self.but1.setFixedHeight(150)
        self.but1.setFont(font_but)
        hbox = QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(self.but1,1,0)
        # self.but2 = PushBut1(self)
        # self.but2.setText("Process")
        # self.but2.setFixedWidth(72)
        # self.but2.setFont(font_but)
        # self.but3 = PushBut1(self)
        # self.but3.setText("")
        # self.but3.setFixedWidth(72)
        # self.but3.setFont(font_but)
        # self.but4 = PushBut1(self)
        # self.but4.setText("")
        # self.but4.setFixedWidth(72)
        # self.but4.setFont(font_but)
        # self.but5 = PushBut1(self)
        # self.but5.setText("")
        # self.but5.setFixedWidth(72)
        # self.but5.setFont(font_but)
        # self.but6 = PushBut1(self)
        # self.but6.setText("")
        # self.but6.setFixedWidth(72)
        # self.but6.setFont(font_but)
        # self.but7 = PushBut1(self)
        # self.but7.setText("")
        # self.but7.setFixedWidth(72)
        # self.but7.setFont(font_but)
        # self.lb1 = QtWidgets.QLabel(self)
        # self.lb1.setFixedWidth(72)
        # self.lb1.setFixedHeight(72)
        self.grid1 = QtWidgets.QGridLayout()
        # self.grid1.addWidget(self.textf, 0, 0, 14, 13)
        self.grid1.addWidget(self.but1, 0, 14, 1, 1)
        # self.grid1.addWidget(self.but2, 1, 14, 1, 1)
        # self.grid1.addWidget(self.but3, 2, 14, 1, 1)
        # self.grid1.addWidget(self.but4, 3, 14, 1, 1)
        # self.grid1.addWidget(self.but5, 4, 14, 1, 1)
        # self.grid1.addWidget(self.but6, 5, 14, 1, 1)
        # self.grid1.addWidget(self.but7, 6, 14, 1, 1)
        # self.grid1.addWidget(self.lb1, 12, 14, 1, 1)
        self.grid1.setContentsMargins(7, 7, 7, 7)
        self.setLayout(self.grid1)
        self.but1.clicked.connect(self.on_but1)
        # self.but2.clicked.connect(self.on_but2)
        
    def on_but1(self):
        # self.textf.setStyleSheet("margin: 1px; padding: 7px; background-color: rgba(1,255,217,100); color: rgba(0,190,255,255);"
        #                          "border-style: solid; border-radius: 3px; border-width: 0.5px; border-color: rgba(0,140,255,255);")
        # filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
        # print('Path file :', filename)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Image Files (*.jpg)", options=options)
        if fileName:
            imagePath = fileName
            # print(imagePath)

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
            imagePath1=imagePath
            input_img = cv2.imread(imagePath1)
            gray_image = cv2.imread(imagePath1, 0)


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
                cv2.imshow('im', result)
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


                

                    # txt = self.textf.toPlainText()
                    # try:
                    #     img = QtGui.QPixmap(txt)
                    #     self.lb1.setPixmap(img.scaledToWidth(72, QtCore.Qt.SmoothTransformation))
                    # except:
                    #     pass

                    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    desktop = QtWidgets.QApplication.desktop()
    resolution = desktop.availableGeometry()
    myapp = PyQtApp()
    myapp.setWindowOpacity(0.95)
    myapp.show()
    myapp.move(resolution.center() - myapp.rect().center())
    sys.exit(app.exec_())
else:
    desktop = QtWidgets.QApplication.desktop()
    resolution = desktop.availableGeometry()
    