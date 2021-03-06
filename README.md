# Sock Sorting with AWS Deeplens

## Table of Contents
- [Overview](#Overview)
- [Use](#Use)
- [My Process](#My-Process)
- [AWS](#AWS)
- [Tips](#Tips)
- [Contact](#Contact)

## Overview

Python based deep learning image classification program to sort socks with AWS Deeplens. The code in this repository includes the training model and lambda function deployed to the Deeplens device. The actual used code was created and ran on AWS Sagemaker and AWS Lambda.

I highly recommend visiting the [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html) website to view the latest documentation, descriptions, and examples.

NOTE: I carried out all of my testing and coding on AWS Sagemaker. I am just consolidating the code here in this repo so it can be shared and viewed by others. Therefore, the time scale of the commits is definitely not realistic. Most of what I was able to copmlete in each commit took days for me to actually test out correctly on Sagemaker.

## Use

This code can be repurposed for any other basic image classification program by inputting different training images, supervised output maps, and adjusting any necessary hyper parameters.

## My Process

I started this project by following [this project](https://simon-aubury.medium.com/sorting-my-socks-with-deep-learning-part-1-1b5651d35f3e). I was able to get it to work as far as the end of the aforementioned article. However, I found that the author trained the model with images of socks in front of a stark white background and also proceeded to put the socks up against a similar white background for the project stream. This made it easy for the model to carryout correct inferences when socks were set against a solid, white background but not when they were set against more complex backgrounds. I wanted to create a more robust model that could handle a variety of backgrounds, which is why I expanded this project to look into concepts like edge detection, contours, morphological transformations, masks, etc. 

The concept behind my first attempt of background subtraction (via edge detection methods) was to focus on trying to extract the sock object from a single image. In this case, the image of a single frame from the project stream. The preprocessing algorithm would take the frame, apply edge detection, find the largest contour (the sock), and then subtract the background outside of that contour. Easier said than done, honestly. First of all, with a complex background, the edge detection function picks up on quite a bit of stuff. Additionally, if your sock has a lot of patterns/colors/color blocking, like mine did, it is not guaranteed that the largest contour will be that of your sock. Such being the case, this did not prove to be a very reliable method either as the outcomes of the edge detection and contour drawing were not consistent in terms of accuracy across different socks and test images. The code for this method can be found in "preprocessing_test.py". If you don't need a robust object detection model then you can definitely just use this method (i.e. less complex backgrounds, solid backgrounds, larger object in reference to the frame, etc.). 

After playing around with Canny edge detection, morphological transformation parameters, and other OpenCV features for days I decided that I should try another approach. Trying to brute force my way through all the different parameter permutations and OpenCV functions was not only taking a really long time, but wasn't working to the same level of efficacy for each sock. I looked online and found another OpenCV background subtractor features that essentially work by trying to distinguish the moving objects from the static background. I tried this next using a sample video of a sock to act as a test/example of a real project stream. 

## AWS

To create and deploy your own image classification project with Deeplens, you must register with AWS. AWS offers a [Free Tier](https://aws.amazon.com/free/) service that allows users to explore more than 85 products and begin building programs/projects on AWS.

First, register your Deeplens device on [AWS Deeplens](https://aws.amazon.com/deeplens/) and connect it to your computer.

Then, go to [AWS S3](https://aws.amazon.com/s3/), which is an object storage platform, and create a bucket for this project. Compile your training images in an ImageRecord format and upload the .rec files to the S3 bucket.

Create a new Notebook Instance in [AWS Sagemaker](https://aws.amazon.com/sagemaker/) (I used a ml.t2.medium instance). This is where you will write and run your training model.

Create a new Lambda Function on [AWS Lambda](https://aws.amazon.com/lambda/). Your lambda function will use your model and run an inference function to try and guess what images your Deeplens is seeing. Visit [this site](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-inference-lambda-create.html) for more information on how to create and publish an AWS Deeplens lambda function.

Finally, from your Deeplens dashboard, create a new project and upload your model and lambda function. You can then deploy this project to your registered Deeplens device. The project stream can either be viewed from the video streaming link on the Devices page or directly from the Ubuntu terminal of the Deeplens itself by running the following code

    mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg

## Tips

I would recommend installing the AWS CLI on your computer. This will assist you with things like uploading content to your S3 buckets.

Additionally, ensure that the mxnet version of your Deeplens is up to date. You can do this by running the following code from the Ubuntu terminal

    sudo pip3 install mxnet==0.12.1

Some of the AWS services you need to use are only available in the US EAST (N. VIRGINIA) region. To ensure you can use all your code in conjunction with each other and deploy them successfully, make sure you are using all AWS services in the US East region.

I made the mistake of leaving my Sagemaker models running for days. DO NOT make this mistake as will be billed for the amount of time you leave your Sagemaker scripts running (RIP wallet). 

Finally, I highly recommend utilizing the IoT console to view MQTT messages from your training model. This will help with debugging. Also, AWS CloudWatch will also come in handy to monitor your lambda function code.

## Contact

I am definitely a beginner in the whole world of AWS, ML, and Python. I warmly welcome any advice or critiques! I am slowly trying to learn as much as I can by reading documentation and viewing other people's code. Definitely reach out via my [GitHub Profile](https://github.com/sophia2798) or my [LinkedIn Profile](https://linkedin.com/in/sophia2798) if you would like to collaborate or talk! 