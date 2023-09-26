import cv2 as cv
import numpy as np
import sys
import ffmpegcv

url = 'rtsp://admin:satech1234@10.10.30.252:10554/udp/av0_0'
np.set_printoptions(formatter={'int':hex})
cap = ffmpegcv.VideoCapture(url)
cap.set(cv.CAP_PROP_FORMAT,-1)
if(cap.get(cv.CAP_PROP_FORMAT) !=-1): print('Unable to activate raw bitstream reads')
iSpsPps = int(cap.get(cv.CAP_PROP_CODEC_EXTRADATA_INDEX))
ret, spsPps = cap.retrieve(flag=iSpsPps)
if(not ret): print('Unable to retrieve parapeter sets')
f = open("out.264", "wb")
spsPps.tofile(f)
for i in range(100):
    ret, encodedFrame = cap.read()
    if(not ret): 
        print('Unable to retrieve encoded frame')
        break
    encodedFrame.tofile(f)