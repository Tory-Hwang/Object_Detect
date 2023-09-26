import cv2
import numpy as np
import subprocess
import json
import sys

import os
os.chdir('D:\YoLo\V3')

def construct_yolo_v3():
    #f=open('coco_names.txt', 'r')
    f=open('yolo_name.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    #https://pjreddie.com/darknet/yolo/
    #model=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    model=cv2.dnn.readNet('yolov3_tiny.weights','yolov3_tiny.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.3:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv2.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects


model,out_layers,class_names=construct_yolo_v3()		# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔


# Use public RTSP Stream for testing
rtsp_url = 'rtsp://admin:satech1234@10.10.30.252:10554/udp/av0_0'
probe_command = ['d:/ffmpeg/bin/ffprobe.exe',
                '-loglevel', 'error',
                '-rtsp_transport', 'udp',  # Force TCP (for testing)]
                '-select_streams', 'v:0',  # Select only video stream 0.
                '-show_entries', 'stream=width,height', # Select only width and height entries
                '-of', 'json', # Get output in JSON format
                rtsp_url]

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

one_frame_size = width*height*3
command = ['d:/ffmpeg/bin/ffmpeg.exe', # Using absolute path for example (in Linux replacing 'C:/ffmpeg/bin/ffmpeg.exe' with 'ffmpeg' supposes to work).
        #'-rtsp_flags', 'listen',   # The "listening" feature is not working (probably because the stream is from the web)
        #'-rtsp_transport', 'tcp',   # Force TCP (for testing)
        #'-max_delay', '30000000',   # 30 seconds (sometimes needed because the stream is from the web).
        #'-r','25',
        '-i', rtsp_url,
        '-f','rawvideo',           # Video format is raw video
        '-pix_fmt', 'bgr24',        # bgr24 pixel format matches OpenCV default pixels format.           
        '-an', 'pipe:']


# Open sub-process that gets in_stream as input and uses stdout as an output PIPE.
ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE)
import time

start=time.time()
n_frame=0

while True:
    # Read width*height*3 bytes from stdout (1 frame)
    raw_frame = ffmpeg_process.stdout.read(one_frame_size)

    if len(raw_frame) != (one_frame_size):
        print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
        break

    # Convert the bytes read into a NumPy array, and reshape it to video frame dimensions
    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    
    res=yolo_detect(frame,model,out_layers)   
    
    for i in range(len(res)):
        x1,y1,x2,y2,confidence,id=res[i]
        text=str(class_names[id])+'%.3f'%confidence
        cv2.rectangle(frame,(x1,y1),(x2,y2),colors[id],2)
        cv2.putText(frame,text,(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,1.5,colors[id],2)
    
    # Show the video frame
    cv2.imshow('RTP', frame)
    n_frame+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end=time.time()
print('처리한 프레임 수=',n_frame,', 경과 시간=',end-start,'\n초당 프레임 수=',n_frame/(end-start))

ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
ffmpeg_process.wait()  # Wait for FFmpeg sub-process to finish

cv2.destroyAllWindows()