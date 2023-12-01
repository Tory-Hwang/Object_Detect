import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

import subprocess
import json
import sys


#cap = cv2.VideoCapture("./Video/video1.mp4")
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

count = 0#
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
#out = cv2.VideoWriter('Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

rtp_url = 'rtsp://admin:satech1234@192.168.0.151:554/udp/av0_0'
        
probe_command = ['d:/ffmpeg/bin/ffprobe.exe',
        '-loglevel', 'error',
        '-rtsp_transport', 'udp',  # Force TCP (for testing)]
        '-select_streams', 'v:0',  # Select only video stream 0.
        '-show_entries', 'stream=width,height', # Select only width and height entries
        '-of', 'json', # Get output in JSON format
        rtp_url]

# Read video width, height using FFprobe:
p0 = subprocess.Popen(probe_command, stdout=subprocess.PIPE)
probe_str = p0.communicate()[0] # Reading content of p0.stdout (output of FFprobe) as string
p0.wait()
probe_dct = json.loads(probe_str) # Convert string from JSON format to dictonary.

# Get width and height from the dictonary
width = probe_dct['streams'][0]['width']
height = probe_dct['streams'][0]['height']
p0.stdout.close()
p0.wait()

one_f_size = width*height*3
command = ['d:/ffmpeg/bin/ffmpeg.exe', # Using absolute path for example (in Linux replacing 'C:/ffmpeg/bin/ffmpeg.exe' with 'ffmpeg' supposes to work).
            #'-rtsp_flags', 'listen',   # The "listening" feature is not working (probably because the stream is from the web)
            #'-rtsp_transport', 'tcp',   # Force TCP (for testing)
            #'-max_delay', '30000000',   # 30 seconds (sometimes needed because the stream is from the web).
            #'-r','25',
            '-i', rtp_url,
            '-f','rawvideo',           # Video format is raw video
            '-pix_fmt', 'bgr24',        # bgr24 pixel format matches OpenCV default pixels format.           
            '-an', 'pipe:']
ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE)   
import time


while True:
    #ret, frame = cap.read()
    raw_frame = ffmpeg_process.stdout.read(one_f_size)
    if len(raw_frame) != (one_f_size):
        print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
        break
    else:
        
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        #cv2.imshow('RTP', frame)
        count += 1

        start_time=time.time()        
        result = list(model.predict(frame, conf=0.35))[0]
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000

        print("Process Time(ms)" , str(total_time_ms))
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100
            label = f'{class_name}{conf}'
            print("Frame N", count, "", x1, y1,x2, y2)
            t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] -3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
           

            #resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        #out.write(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (0, 255, 0)
        cv2.putText(frame, "Predict Time: {:.2f} ms".format(total_time_ms), (10, 100), font, font_scale, font_color, lineType = cv2.LINE_AA)

        cv2.imshow("Frame", frame)
       
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break   
       
        
#out.release()
#cap.release()
ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
ffmpeg_process.wait()  # Wait for FFmpeg sub-process to finish

cv2.destroyAllWindows()