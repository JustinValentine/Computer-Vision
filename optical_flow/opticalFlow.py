from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time 
import math


def temporal_derivative(img_old, img, count=True):
    """ Calculates the signed difference of two sequential frames
    input:  img_old  nxn image at time t
            img      nxn image at time t+1
            count:   boolean flag
    return: image_diff: the thresholded difference of img and img_old
            num_nonzero: number of nonzero pixels 
    """
    # calculate image difference  
    image_diff = img_old - img

    # threshold the result
    threshold = 15 
    image_diff[np.abs(image_diff) < threshold] = 0

    if count:
        # calculate number of nonzero pixels
        num_nonzero = np.count_nonzero(image_diff)
        return image_diff, num_nonzero

    else: 
        return image_diff


def image_gradient(img, Flatten=True):
    """ Calculates the image gradients w.r.t the x-axes and the y-axes
   input: img          nxn image at time t     
          Flatten      boolean flag
   return:
          image_dir_x  x shifted image difference (if Flatten the n^2x1 vector)
          image_dir_y  y shifted image difference (if Flatten the n^2x1 vector)
          I_grad       col 1 is flattened gradinet w.r.t x 
                       col 2 is flattened gradinet w.r.t y (n^2x2)
    """
    # approximate image gradient
    image_dir_x = img - np.roll(img, 1, axis=1)
    image_dir_y = img - np.roll(img, 1, axis=0)

    # set wrap around pixles to 0 
    image_dir_x[:, :1] = 0
    image_dir_y[:1, :] = 0

    if not Flatten:
        return image_dir_x, image_dir_y

    # convert to column vector 
    image_dir_x = image_dir_x.flatten()
    image_dir_y = image_dir_y.flatten()
    
    # combine derivatives to form image gradient
    I_grad = np.column_stack((image_dir_x, image_dir_y))

    return I_grad


def optical_flow(img0, img1, size=8):
    """ calulates the image flow vector field
    input: img0     nxn image at time t
           img1     nxn image at time t+1
           size     subdivide image into grid, each cell has side lengths size 
    return:
           flow     an array containing the x-position, y-position, x-direction, y-direction,
                    of all vectors in the flow vector field.
                    format [x-position,y-position,x-direction,y-direction] where each
                    element is a n^2x1 vector
    """

    # find dimensions of the image
    w, h = img0.shape

    # find number of grid cells 
    gride_size = w//size

    # calculate the image gradient w.r.t both x and y directions
    image_dir_x, image_dir_y = image_gradient(img0, Flatten=False)

    # vector info for quiver
    x_pos, y_pos = [], []
    x_direct, y_direct = [], []
    test_eigenval = []

    # loop over image grid 
    for i in range(gride_size):
        for j in range(gride_size):
            # get grid cell for images at time t and t+1 
            cell0 = img0[i*size:(i+1)*size, j*size:(j+1)*size]
            cell1 = img1[i*size:(i+1)*size, j*size:(j+1)*size]

            # calculate the temporal derivative of the cell at time t
            time_dir_cell, count = temporal_derivative(cell0, cell1)

            # skip the rest if all 0 in temporal derivative
            if count == 0:
                x_direct.append(0)
                y_direct.append(0)
                x_pos.append(j*size + size//2)
                y_pos.append(i*size + size//2)
                continue
            
            # convert temporal derivative to column vector and multiply by -1 
            time_dir_cell = -1 * time_dir_cell.flatten()

            # get the image gradient w.r.t x and y for the grid cell 
            image_dir_x_2 = image_dir_x[i*size:(i+1)*size, j*size:(j+1)*size].flatten()
            image_dir_y_2 = image_dir_y[i*size:(i+1)*size, j*size:(j+1)*size].flatten()

            # concatenate the gradients
            I_grad = np.column_stack((image_dir_x_2, image_dir_y_2))

            H = np.transpose(I_grad) @ I_grad
            eigenval, eigenvec = np.linalg.eig(H)
            eigen_ratio = eigenval[0]/eigenval[1]
            test_eigenval.append(eigenval[0]/eigenval[1])

            if eigen_ratio < 0.08 or eigen_ratio > 2.6:
                x_direct.append(0)
                y_direct.append(0)
                x_pos.append(j*size + size//2)
                y_pos.append(i*size + size//2)
                continue

            # solve the optical flow equation by finding the least square solution
            delta_x_y = np.linalg.lstsq(I_grad, time_dir_cell)[0]
            length = np.linalg.norm(delta_x_y)

            if length >= 1:
                delta_x_y /= length

            x_direct.append(delta_x_y[0])
            y_direct.append(delta_x_y[1])
            x_pos.append(j*size + size//2)
            y_pos.append(i*size + size//2)

    if len(test_eigenval) != 0:
        print("10", np.percentile(test_eigenval, 10))
        print("90", np.percentile(test_eigenval, 90))

    flow = [x_pos, y_pos, x_direct, y_direct]
    return flow


def makeVideo():
    """
    convert the image sequence in output directory into a mp4 named 'output.mp4'
    """
    # read image frame
    frame = cv.imread("output/0.png")

    # get dimensions of the frame
    height, width, layers = frame.shape

    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video = cv.VideoWriter("output.mp4", fourcc, 30, (width, height))

    for i in range(len(list(Path('output').iterdir()))) :
        img = cv.imread("output/{}.png".format(i))
        video.write(cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))


def GuassPyramid(img, num=4):
    """ applies gaussian subsampling to an image num times """
    for _ in range(num):
        img = cv.pyrDown(img)

    return img 


def DisplayAll(dt, neg_dt, dx, dy, img, flow, save=False, frame=-1):
    """ - Displays relevant image information for calculating optical flow.
        - Saves frames to output folder  
    """
    images = [dt, neg_dt, dx, dy]
    names = ['Temporal Derivative', 'neg Temporal Derivative', 'Image Gradient x', 'Image Gradient y']

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
    plt.quiver(flow[0], flow[1], flow[2], flow[3],
               color='r', scale=10)
    plt.axis('off')
    plt.title("optic flow lstsq")

    if save: 
        if frame >= 0:
            plt.savefig('output/{}.png'.format(frame))

    plt.show()
    plt.clf()
    

def saveOpticalFlow(flow, img, frame):
    """ saves frames to output folder """

    plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)
    plt.quiver(flow[0], flow[1], flow[2], flow[3],
               color='b', scale=12)
    plt.axis('off')
    plt.title("Optical Flow ")
    plt.savefig('output/{}.png'.format(frame))
    plt.clf()


def main():
    frame_new = cv.imread("input/0.png", 0) 
    w, h = frame_new.shape
    step = 1

    for i in range(100): 
        # load frames from input directory as grayscale 
        frame_old = frame_new 
        frame_new = cv.imread("input/{}.png".format((i+1)*step), 0) 

        # convert pixel values to type double 
        frame_old, frame_new = np.double(frame_old), np.double(frame_new)

        # applies gaussian subsampling to frames
        reduced_image_size = 8 # new frame pixel length 2^reduced_image_size
        layer0 = GuassPyramid(frame_old, np.abs(reduced_image_size - math.floor(math.log2(w))))
        layer1 = GuassPyramid(frame_new, np.abs(reduced_image_size - math.floor(math.log2(w))))

        # calculates components of optical flow equation to be displayed
        # dt = temporal_derivative(layer0, layer1, False)
        # dx, dy = image_gradient(layer0, Flatten=False)

        # calulates the image flow vector field
        flow = optical_flow(layer0, layer1)

        #DisplayAll(dt, -1*dt, dx, dy, layer0, flow, save=True, frame=i)
        saveOpticalFlow(flow, layer0, frame=i)

        print(i)
    
    makeVideo()

if __name__ == "__main__":
    main()