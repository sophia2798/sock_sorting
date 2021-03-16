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
cv2.imwrite(resized_file_name, newimg) # this is an optional step, but I will be saving a new image each time a new preprocessing step takes place