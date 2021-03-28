import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance

'''
For some reason, Jupyter Notebook/SageMaker doesn't like the cv.imshow() method. It seems to freeze the notebook and kill the kernel. I found a solution online that basically uses the matplotlib module's pyplot function to display the desired image(s) on a graph.

It should be noted that images are loaded in BGR by default, so you will want to convert them to RGB using cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    fgmask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB)

    # add contrast to mask
    fgmask_format = Image.fromarray(fgmask_rgb)
    contrast = 4
    enhance_contrast = ImageEnhance.Contrast(fgmask_format)
    contrasted = enhance_contrast.enhance(contrast)
    contrasted_arr = np.array(contrasted)

    # use canny edge detection
    erode = cv2.erode(contrasted_arr,kernel,iterations=5)
    dilate = cv2.dilate(erode,kernel,iterations=1)
    edges = cv2.Canny(dilate, 10, 200, apertureSize=3)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    '''
    NEXT, I THINK I AM GOING TO TRY TO ADD A WHITE COLOR MASK. I TRIED THIS EARLIER BUT IT DIDN'T WORK BECAUSE THE SHAPE OF THE FGMASK ARRAY IS HUGE AND THE COLOR ARRAYS YOU CAN SPECIFY WHEN APPLYING A MASK ARE (3,1) SO I NEED TO FIND A WAY TO RESHAPE MY FGMASK
    '''

    # conditional to check if there are any contours in the frame
    if hierarhcy is None:
        plt.imshow(cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB))
    else:
        sigcontour = findSignificantContour(edges)
        copy = np.copy(fgmask)
        cv2.drawContours(copy, sigcontour, -1, (0,255,0), 2, cv2.LINE_AA, maxLevel=1)
        plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))

    plt.show()

# the code below is what you could use if cv2.imshow() function works in your python IDE
#     cv2.imshow('frame', fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
        
cap.release()
# cv2.destroyAllWindows()