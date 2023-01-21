from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time 
import math


def temporal_derivative(img, img_old, count=True):
    # Calculate temporal derivative 
    image_diff = np.abs(img - img_old)

    # Threshold result
    threshold = 90
    image_diff[image_diff < threshold] = 0
    #image_diff[image_diff >= threshold] = 255

    if count:
        return image_diff, np.count_nonzero(image_diff)

    else: return image_diff


def image_gradient(img, Flatten=True):
    # approximate image gradient np.roll
    image_dir_x = img - np.roll(img, 1, axis=1)
    image_dir_y = img - np.roll(img, 1, axis=0)
    image_dir_x[:, :1] = 0
    image_dir_y[:1, :] = 0

    if not Flatten:
        return image_dir_x, image_dir_y

    # convert to column vector 
    image_dir_x = image_dir_x.flatten()
    image_dir_y = image_dir_y.flatten()
    
    # combine dirivatives to form image gradient 
    I_grad = np.column_stack((image_dir_x, image_dir_y))

    return I_grad

def check_vec(x_vec, y_vec):
    pass


def optical_flow(img1, img2, size=8, pinv=False):
    w, h = img1.shape
    gride_size = w//size

    image_dir_x, image_dir_y = image_gradient(img1, Flatten=False)

    # info for quiver
    x_pos, y_pos = [], []
    x_direct, y_direct = [], []

    for i in range(gride_size):
        for j in range(gride_size):
            # get image cell at t and t+1 
            cell1 = img1[i*size:(i+1)*size, j*size:(j+1)*size]
            cell2 = img2[i*size:(i+1)*size, j*size:(j+1)*size]

            # Calculate the temporal derivative of the cell at time t
            time_dir_cell, count = temporal_derivative(cell1, cell2)
            
            # convert temporal derivative to column vector 
            time_dir_cell = -1 * time_dir_cell.flatten()

            image_dir_x_2 = image_dir_x[i*size:(i+1)*size, j*size:(j+1)*size].flatten()
            image_dir_y_2 = image_dir_y[i*size:(i+1)*size, j*size:(j+1)*size].flatten()
            I_grad = np.column_stack((image_dir_x_2, image_dir_y_2))

            # least square solution 
            delta_x_y = np.linalg.lstsq(I_grad, time_dir_cell)[0]

            if pinv:
                delta_x_y = np.linalg.pinv(I_grad) @ time_dir_cell

            x_direct.append(delta_x_y[0])
            y_direct.append(delta_x_y[1])
            x_pos.append(j*size + size//2)
            y_pos.append(i*size + size//2)

    check_vec(x_direct, y_direct)

    return [x_pos, y_pos, x_direct, y_direct] 


def makeVideo():
    images = list(Path('output').iterdir())

    frame = cv.imread("output/0.png")
    height, width, layers = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video = cv.VideoWriter("output.mp4", fourcc, 30, (width, height))

    for i in range(len(images)) :
        img = cv.imread("output/{}.png".format(i))
        video.write(cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))


def GuassPyramid(img, size=4):
    for _ in range(size):
        img = cv.pyrDown(img)

    return img 


def DisplayAll(dt, neg_dt, dx, dy, img, flow_p, flow, save=False, frame=-1):
    images = [dt, neg_dt, dx, dy]
    names = ['dt', 'neg_dt', 'dx', 'dy']

    # create figure
    fig = plt.figure()

    # setting values to rows and column variables
    rows, columns = 2, 3

    for i in range(4):
        # Adds a subplot 
        fig.add_subplot(rows, columns, i+1)
        
        # showing image
        plt.imshow(np.int16(images[i]), cmap='gray', vmin=-255, vmax=255)
        plt.axis('off')
        plt.title(names[i])

    fig.add_subplot(rows, columns, 5)
    plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)
    plt.quiver(flow_p[0], flow_p[1], flow_p[2], flow_p[3], 
               pivot='mid', color='r', scale=10)
    plt.axis('off')
    plt.title("optic flow pinv")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)
    plt.quiver(flow[0], flow[1], flow[2], flow[3],
               color='r', scale=10)
    plt.axis('off')
    plt.title("optic flow lstsq")

    if frame >= 0:
        plt.savefig('output/{}.png'.format(frame))

    #plt.show()
    plt.clf()
    

def main():
    frame_new = cv.imread("input/0.png", 0) 
    w, h = frame_new.shape
    step = 1

    for i in range(50): 
        # Load frames 
        frame_old = frame_new 
        frame_new = cv.imread("input/{}.png".format((i+1)*step), 0) 
        frame_old, frame_new = np.double(frame_old), np.double(frame_new)

        # gaussian pyramid image processing
        reduced_image_size = 8 # new size 2^reduced_image_size
        layer0 = GuassPyramid(frame_old, np.abs(reduced_image_size - math.floor(math.log2(w))))
        layer1 = GuassPyramid(frame_new, np.abs(reduced_image_size - math.floor(math.log2(w))))

        dt = temporal_derivative(layer0, layer1, False)
        dx, dy = image_gradient(layer0, Flatten=False)
        flow_p = optical_flow(layer0, layer1, pinv=True)
        flow = optical_flow(layer0, layer1)

        DisplayAll(dt, -1*dt, dx, dy, layer0, flow_p, flow, save=True, frame=i)

        print(i)
    
    makeVideo()

if __name__ == "__main__":
    main()