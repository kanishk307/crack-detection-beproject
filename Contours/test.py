import cv2
import numpy as np

src = cv2.imread("a3.jpg", 1) #image path dalo
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # grayscale, jarurat naiye but incase image hoga toh rehnediya. vaise abhi binary hi hai so no issues
blur = cv2.blur(gray, (3, 3)) # iska bhi vaise jarurat naiye par better result deray thoda. not noteworthy
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

cv2.imshow('draw',drawing)
 
# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green contour
    color = (255, 0, 0) # blue contour
    # ek ek karke draw green contour
    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw blue convex 
    cv2.drawContours(drawing, hull, i, color, 1, 8)
    # print(cv2.contourArea(hull)) #IDHAR KAAM KARNA HAI



cv2.imshow('draw',drawing)
cv2.waitKey(0)