# THE FOLLOWING CODE CAN BE USED IN YOUR SAGEMAKER NOTEBOOK TO TEST AN UPLOADED IMAGE TO YOUR S3 BUCKET AGAINST YOUR MODEL

import os
import urllib.request
import boto3
from IPython.display import Image
import cv2
import json
import numpy as np 

# input the S3 bucket you are using for this project and the file path for a folder and file that contains your uploaded test image
test_image_bucket = 'deeplens-sagemaker-socksortingeast'
test_image_name = 'testimages/image0.jpeg'

tmp_file_name = 'tmp-test-image-jpg'
resized_file_name = 'resized-test-image.jpg'
s3 = boto3.client('s3')
with open(tmp_file_name, 'wb') as f:
    s3.download_fileobj(test_image_bucket, test_image_name, f)

# width
W = 500
oriimg = cv2.imread(tmp_file_name)
height, width, depth = oriimg.shape
# scale the image
imgScale = W/width
newX,newY = oriimg.shape[1].imgScale, oriimg.shape[0]*imgScale
newimg = cv2.resize(oriimg, (int(newX),int(newY)))
cv2.imwrite(resized_file_name, newimg)

with open(resized_file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)

result = json.loads(ic_classifier.predict(payload, initial_args={'ContentType': 'application/x-image'}))
# find the index of the class that matches the test image with the highest probability
index = np.argmax(result)
# input your own output categories
object_categories = ['BlueStripes', 'DarkGray', 'IronMan']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))
print()
print(result)
print(ic._current_job_name)
Image(resized_file_name)