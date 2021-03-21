import os
from PIL import ImageEnhance
from PIL import Image as Im
import boto3
from IPython.display import Image
import cv2
import json
import numpy as np

test_image_name = 'IMG_0177.JPEG' 
# use the name of whatever test image you would like to use; you can import directly into the sagemaker directory or from an s3 bucket

# APPLY A SLIGHT GAUSSIAN BLUR

image_org = cv2.imread(test_image_name)
gauss_blurred = cv2.GaussianBlur(image_org, (5,5), 0)

# RESIZE THE IMAGE (for testingp purposes so output is easier to see on the screen AND/OR to scale the frame from the DeepLens to that of your training images)

height, width, depth = gauss_blurred.shape
imgScale = 0.17 # this number can be changed depending on your preferences
newX,newY = gauss_blurred.shape[1]*imgScale, gauss_blurred.shape[0]*imgScale
newimg = cv2.resize(gauss_blurred, (int(newX),int(newY)))
resized_file_name = 'preprocessing_resize.jpg'
cv2.imwrite(resized_file_name, newimage) # this is an optional step, but I will be saving a new image each time a new preprocessing step takes place

with open(resized_file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)

# APPLY A CONTRAST TO THE IMAGE

newimg = Im.fromarray(newimage) # the PIL.Image.fromarray() and np.array() function online 38, are important to note. some functions/preprocessing techniques require the image to be in PIL format while some require it to be in the form of a numpy array. the two functions mentioned can be used to convert an image between the two formats.
enh_con = ImageEnhance.Contrast(newimg)
contrast = 3.01 
img_contrasted = enh_con.enhance(contrast)
image = img_contrasted
image = np.array(image)
contrast_file_name = 'contrast-image.jpg'
cv2.imwrite(contrast_file_name, image)

# APPLY MORPHOLOGICAL TRANSFORMATIONS AND CANNY EDGE DETECTION

'''
this is one of the main parts of the code that i had to play around with A LOT. just trying out different settings and figuring out what best suited my needs. depending on what object you are trying to detect, you might want to use different settings (i.e. iterations, aperature size, etc.). i started out only applying the canny edge detection, but found that the dilation and erosion transformations helped a great deal in singling out the major/important edges.
'''

MORPH = 9
img = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY) # many of the preprocessing steps we will end up taking (i.e. threshold) require the image to be on the grayscale
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH,MORPH)) # this function allows you to input shape and size and will output the desired kernel. this is so you don't have to manually create the structuring elements.
dilated = cv2.dilate(img,kernel,iterations=1)
eroded = cv2.erode(dilated,kernel,iterations=3)
edges = cv2.Canny(eroded,0,100,aperatureSize=3) # aperature size has to be an odd integer between 3-7
canny_file_name = 'edge-raw.jpg'
cv2.imwrite(canny_file_name,edges)

# OPTION: you can use 'Image(file_name)' at any point to view a displayed output of your image at any point in the preprocessing 

# FILTER OUT SALT AND PEPPER NOISE

# we will use a median filter, which computes the median of all the pixels within a kernel window and replaces the central pixel with the median value.

def SaltPepperNoise(edgeImg):
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg,1)
    while not np.array_equal(lastMedian,median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0
        count += 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg,1)

edges_ = np.asarray(edges, np.uint8) # this assigns the data type as an 8-bit unsigned integer, which ensures the image will be displayed as is
SaltPepperNoise(edges_)
cv2.imwrite('edge.jpg', edges_)

# FIND SIGNIFICANT/LARGEST CONTOUR

'''
this was the other part of the preprocessing code that took up a lot of my time. i tried a couple of different functions to identify the largest contour. i started off by just singling out the largest contour and then also added a rectangular boundary/contour around that contour later.
'''

ret, thresh = cv2.threshold(edges_, 127, 255, 0)

def findSignificantContours(edgeImg, img):
    contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#  this was the first piece of code i tried...
#     level1Meta = []
#     for contourIndex, tupl in enumerate(hierarchy[0]):
#         if tupl[3] == -1:
#             tupl = np.insert(tupl.copy(), 0, [contourIndex])
#             level1Meta.append(tupl)
            
#     contoursWithArea = []
#     for tupl in level1Meta:
#         contourIndex = tupl[0]
#         contour = contours[contourIndex]
#         area = cv2.contourArea(contour)
#         contoursWithArea.append([contour, area, contourIndex])
        
#     contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
#     largestContour = contoursWithArea[0][0]
#     return largestContour
    
    if len(contours) != 0:
        cv2.drawContours(img, contours, -1, 255, 1) # -1 in the third argument space means that all contours will be drawn
        maxContour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, c, -1, (60,40,220),3)
        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
        return img

# SCALING CONTOUR

'''
this is an optional step that you may or may not want/need. the dilation and erosion transformations change the size of your contour, so i included a short scaling function to help scale it back to its original size. that way when the contour is applied back to the original image for background subtraction it is to scale.
'''

def scaleContour(cont, scale):
    M = cv2.moments(cont)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    cont_norm = cont - [cx,cy]
    cont_scaled = cont_norm*scale
    cont_scaled += [cx,cy]
    cont_scaled = cont_scaled.astype(np.int32) # np.int32 is the same as np.uint8

    return cont_scaled

# APPLY SIGNIFICANT CONTOUR FUNCTION TO ORIGINAL IMAGE

contourImg = np.copy(newimage)
output = findSignificantContours(edges_, contourImg)
cv2.imwrite('contour.jpg', output)