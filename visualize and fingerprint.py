##########################################################
#                                                        #
#   An implimentation of k-Nearest Neighbor for audio    #
#   classification                                       #
#                                                        #
##########################################################

import sys
import pyaudio
import colorsys
import numpy as np
import time
import cv2

CHUNK = 512 # number of data points to read at a time
RATE = 10000 # time resolution of the recording device (Hz)... I recommend the use of 10K - 20K
FRAMES = 100    #adjust this the change the length of time you wish to record
LAYERS = 3      #the program has three color layers
THRESH = 0.1    #If you can get the thresh to work, good for you!

def hsv2rgb(d): #Allows us to convert frequencues and intensity to color
    dat = []
    for h in d:
        dat.append(list(int(i * 255) for i in colorsys.hsv_to_rgb(h,1.0,1.0)))
    return dat

#print hsv2rgb(0.83)
#sys.exit()
kernel = np.ones((2,2),np.float32)/CHUNK

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True, frames_per_buffer=CHUNK)

img = np.zeros((FRAMES, CHUNK, LAYERS))
#print img[0]

start = time.clock()
for i in range(FRAMES):
    data = np.fromstring(stream.read(CHUNK),dtype=np.float32)
    data = (np.fft.fftn(data))
    data /= np.max(np.abs(data),axis=0)
    #data = data[:] > THRESH
    rgb = [list(j) for j in hsv2rgb(data)]
    for j in xrange(len(data)):
        img[i][j][:3] = rgb[j][:3]
print time.clock() - start
#img = np.array(img * 255, dtype=np.uint8)  #Good luck using this line
#threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)    #Throws an error that breaks the program
img = cv2.filter2D(img,-1,kernel)
cv2.imshow('ImageWindow',img)
cv2.waitKey(0)

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()
