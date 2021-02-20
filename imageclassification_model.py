%%time
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import image_uris
import os
import urllib.request
import boto3

role = get_execution_role
sess = sagemaker.Session()
# input your s3 bucket name
bucket = 'deeplens-sagemaker-socksortingeast'
prefix = 'ic-transfer-learning'

training_image = image_uris.retrieve('image-classification', sess.boto_region_name, version="latest")
print(training_image)

# function to upload files to your S3 bucket
def upload_to_s3(channel, file):
    # boto3 is a Python SDK for AWS that lets you manipulate AWS resources directly from the script
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

s3 = boto3.client('s3')

# you should have 2 .rec files -> one validation and one training
# input the file names of your .rec files in place of the ones below
with open('Astro_val.rec', 'wb') as f:
    s3.download_fileobj(bucket, 'Astro_val.rec', f)

with open('Astro_train.rec', 'wb') as f:
    s3.download_fileobj(bucket, 'Astro_train.rec', f)

upload_to_s3('validation', 'Astro_val.rec')
upload_to_s3('training', 'Astro_train.rec')

# create a training and validation channel, upload the files to the appropriate channels
s3train = 's3://{}/{}/train/'.format(bucket,prefix)
s3validation = 's3://{}/{}/validation/'.format(bucket,prefix)

!aws s3 cp Astro_train.rec $s3train
!aws s3 cp Astro_val.rec $s3validation

# at this point, you should have the .rec files in the root of your deeplens bucket, a folder with the name of your prefix variable on line 13, inside that folder a 'train' and 'validation' channel with the appropriate .rec files in each

