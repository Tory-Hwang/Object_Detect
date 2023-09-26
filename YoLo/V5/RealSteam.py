import cv2
import numpy as np
import subprocess
import json
import sys


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

    # Show the video frame
    cv2.imshow('RTP', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    n_frame+=1

end=time.time()
print('처리한 프레임 수=',n_frame,', 경과 시간=',end-start,'\n초당 프레임 수=',n_frame/(end-start))

ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
ffmpeg_process.wait()  # Wait for FFmpeg sub-process to finish

cv2.destroyAllWindows()