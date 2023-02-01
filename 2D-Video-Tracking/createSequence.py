import cv2 as cv
import math 

vidcap = cv.VideoCapture('test3.mp4')
success, frame = vidcap.read()
frame  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

print(frame.shape)

# converts frame to size nxn where n is a power of 2
w, h = frame.shape 
size = 2 ** math.floor(math.log2(min(w, h)))
print(size)

# cv.imwrite("input/{}.png".format(0), frame[:size, :size])
cv.imwrite("input/{}.png".format(0), frame)

count = 1
success, frame = vidcap.read()

while success:
    frame  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #cv.imwrite("input/{}.png".format(count), frame[:size, :size])
    cv.imwrite("input/{}.png".format(count), frame)
    success, frame = vidcap.read()
    count += 1

print("done")