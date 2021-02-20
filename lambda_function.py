import json
import awscam
import mo
import cv2
import greengrasssdk
import os
import numpy as np
import math
from local_display import LocalDisplay

client = greengrasssdk.client('iot-data')
# you can find your iot_topic string on your DeepLens Devices page under "Project Output"
iot_topic = '$aws/things/YOUR_IOT_TOPIC/infer'
client.publish(topic=iot_topic, payload='At start of lambda function')

def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return

def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    client.publish(topic=iot_topic, payload='Start of run loop...')
    
    local_display = LocalDisplay('480p')
    local_display.start()
    
    input_height = 512
    input_width = 512
    model_name = 'image-classification'
    model_type = 'classification'
    output_map = {0: 'Blue Stripes', 1:'Dark Gray', 2:'Iron Man'}
    client.publish(topic=iot_topic, payload='Optimizing model...')
    # your epoch number will be the number at the end of the params file in the /opt/awscam/artifacts/ file path of your DeepLens
    error, model_path = mo.optimize(model_name,input_width,input_height, aux_inputs={'--epoch': 1})
    client.publish(topic=iot_topic, payload=str(error))
    # Load the model here
    mcfg = {"GPU": 1}
    model = awscam.Model(model_path, mcfg)
    client.publish(topic=iot_topic, payload='Custom object detection model loaded')
    while True:
        # Get a frame from the video stream
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")
        frame_resize = cv2.resize(frame, (input_height, input_width))
        predictions = model.doInference(frame_resize)
        parsed_inference_results = model.parseResult(model_type, predictions)
        
        k = 2
        
        top_k = parsed_inference_results[model_type][0:k]
        print(top_k)
        sock_label = output_map[top_k[0]['label']]
        sock_prob = top_k[0]['prob']*100
        json_payload = {"image" : sock_label, "probability": sock_prob}
        # this is what will be shown on the video stream, personalize it to whatever you like!
        msg_screen = 'ASTRO SAYS {} {:.0f}%'.format(sock_label, sock_prob)
        cv2.putText(frame, msg_screen, (20,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 12)
        local_display.set_frame_data(frame)
        client.publish(topic=iot_topic, payload=json.dumps(json_payload))
        cloud_output = {}
        for obj in top_k:
            cloud_output[output_map[obj['label']]] = obj['prob']
        client.publish(topic=iot_topic, payload=json.dumps(cloud_output))

# run the function
infinite_infer_run()