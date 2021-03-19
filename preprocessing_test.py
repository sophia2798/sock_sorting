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

# RESIZE THE IMAGE (for testingp purposes so output is easier to see on the screen)

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

# this is one of the main parts of the code that i had to play around with A LOT. just trying out different settings and figuring out what best suited my needs. depending on what object you are trying to detect, you might want to use different settings (i.e. iterations, aperature size, etc.). i started out only applying the canny edge detection, but found that the dilation and erosion transformations helped a great deal in singling out the major/important edges.

MORPH = 9
img = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY) # many of the preprocessing steps we will end up taking (i.e. threshold) require the image to be on the grayscale
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH,MORPH)) # this function allows you to input shape and size and will output the desired kernel. this is so you don't have to manually create the structuring elements.
dilated = cv2.dilate(img,kernel,iterations=1)
eroded = cv2.erode(dilated,kernel,iterations=3)
edges = cv2.Canny(eroded,50,200,aperatureSize=3) # aperature size has to be an odd integer between 3-7
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