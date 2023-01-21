import cv2 as cv
import math 

vidcap = cv.VideoCapture('test6.mp4')
success, frame = vidcap.read()
frame  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

print(frame.shape)

w, h = frame.shape 
size = 2 ** math.floor(math.log2(min(w, h)))
print(size)

cv.imwrite("input/{}.png".format(0), frame[:size, :size])
count = 1
success, frame = vidcap.read()

while success:
    frame  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imwrite("input/{}.png".format(count), frame[:size, :size])
    success, frame = vidcap.read()
    count += 1

print("done")