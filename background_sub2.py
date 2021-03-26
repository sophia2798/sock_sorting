import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

'''
For some reason, Jupyter Notebook/SageMaker doesn't like the cv.imshow() method. It seems to freeze the notebook and kill the kernel. I found a solution online that basically uses the matplotlib module's pyplot function to display the desired image(s) on a graph.
'''

test_video_name = 'sock_example.avi'

cap = cv2.VideoCapture(test_video_name)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

# same function from preprocessing_test.py
def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    level1Meta = []
    for contourIndex, tupl in enumerate(hierarhcy[0]):
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)

    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # the image is loaded in BGR by default, want to convert it to RGB
    plt.imshow(cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB))
    plt.show()

# the code below is what you could use if cv2.imshow() function works in your python IDE
#     cv2.imshow('frame', fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
        
cap.release()
# cv2.destroyAllWindows()