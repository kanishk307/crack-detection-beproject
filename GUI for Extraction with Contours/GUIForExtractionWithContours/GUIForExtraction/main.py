import cv2
import math
import numpy as np
import scipy.ndimage
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.properties import ObjectProperty
from PIL import Image


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

    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    return result


def contour_function(src):
    areasum=0
    height, width = src.shape
    frameSize = width * height
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # grayscale, jarurat naiye but incase image hoga toh rehnediya. vaise abhi binary hi hai so no issues
    blur = cv2.blur(src, (3, 3)) # iska bhi vaise jarurat naiye par better result deray thoda. not noteworthy
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # hull array
    hull = []
    
    # no of points for each contor
    for i in range(len(contours)):
        # convex hull obj for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # empty kala dabba
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green contour
        color = (255, 0, 0) # blue contour
        # ek ek karke draw green contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw blue convex 
        cv2.drawContours(drawing, hull, i, color, 1, 8)
        cnt = hull[i]
        M=cv2.moments(cnt)
        # print(M)
        area = cv2.contourArea(cnt)
        areasum=areasum+area
        intensity =  areasum * 100 /frameSize 

    return drawing,areasum,intensity
    



class RootWidget(TabbedPanel):
    manager = ObjectProperty(None)
    img = ObjectProperty(None)
    img3 = ObjectProperty(None)
    img4 = ObjectProperty(None)
    lab = ObjectProperty(None)
    lab1 = ObjectProperty(None)

    # def on_touch_up(self, touch):
    #     self.lab.text = 'Areasum'
    #     if not self.img3.collide_point(*touch.pos):
    #         return True
    #     else:
    #         self.lab.text = 'Pos: (%d,%d)' % (touch.x, touch.y)
    #         return True

    def switch_to(self, header):
        self.manager.current = header.screen

        self.current_tab.state = "normal"
        header.state = 'down'
        self._current_tab = header

    def select_to(self, *args):
        try:
            print(args[1][0])
            original_image = cv2.imread(args[1][0])
            image_gray = cv2.imread(args[1][0],0)
            cv2.imwrite('C:/GUIForExtraction/original_im.jpg',original_image)
            result = main_function(image_gray)
            cv2.imwrite('C:/GUIForExtraction/processed_im.jpg', result)

            result2 = main_function(image_gray)
            cv2.imwrite('C:/GUIForExtraction/processed_im_c.jpg',result2)
            
            contour_img,areasum,intensity=contour_function(result2)
            
            print(areasum)
            print(intensity)


            cv2.imwrite('C:/GUIForExtraction/processed_im_c.jpg',contour_img)
           
            self.img3.source = './processed_im.jpg'
            self.img4.source = './processed_im_c.jpg'
            self.img.source = './original_im.jpg'

            self.lab.text = 'Crack Area : (%d)' % (areasum)
            self.lab1.text = "Crack Intensity : (%d)" % (intensity)

            
            

            # print(areasum)
            # count_black = np.sum(result==0)
            # count_white = np.sum(result == 255)
            # print('Number of white pixels : ', count_white)
            # print('Number of black pixels : ', count_black)
            # intensity = count_white*100/count_black
            # print('Intensity : ',intensity)
            self.img.reload()
            self.img3.reload()
            self.img4.reload()
        except:
            pass

    # def update_touch_label(self, label, touch):
    #     label.text = 'Pos:(%d, %d)' % (touch.x, touch.y)
    #     label.texture_update()
    #     label.pos = touch.pos
    #     label.size = label.texture_size[0] + 20, label.texture_size[1] + 20


class TestApp(App):
    title = 'Crack Feature Extraction'

    def build(self):
        return RootWidget()

    def on_pause(self):
        return True


if __name__ == '__main__':
    TestApp().run()
