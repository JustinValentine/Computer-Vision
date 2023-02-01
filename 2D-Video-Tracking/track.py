from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time 
import math


def calculate_image_diff(img0, img1, count=True):
    """ Calculates the signed difference of the template and the current frame
    input:  img0     nxn image 
            img1     nxn image 
            count:   boolean flag
    return: image_diff: the thresholded difference of img0 and img1
            num_nonzero: number of nonzero pixels 
    """
    # calculate image difference  
    image_diff = img0 - img1

    # threshold the result
    threshold = 10 
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


def track_image(T, I, template, box, p):
    """ calulates the image flow vector field
    input: T            nxn image of the inital region we wont to track
           I            nxn image at time t
           template     the rectange around the inital region  
    return:
           delta_p      the movment of the reigon we want to track 
    """

    # calculate the image gradient w.r.t both x and y directions
    image_dir_x, image_dir_y = image_gradient(I, Flatten=False)

    # crop image frames, and convert to column vectors 
    image_dir_x = image_dir_x[box[1]:box[3]+box[1], 
                              box[0]:box[2]+box[0]].flatten()
    
    image_dir_y = image_dir_y[box[1]:box[3]+box[1], 
                              box[0]:box[2]+box[0]].flatten()

    # crop image frames 
    T = T[template[1]:template[3]+template[1], 
          template[0]:template[2]+template[0]]
    
    I = I[box[1]:box[3]+box[1], 
          box[0]:box[2]+box[0]]
    
    # calculate the temporal derivative of the cell at time t
    img_error, count = calculate_image_diff(T, I)

    # convert to column vectors 
    T = T.flatten()
    I = I.flatten()

    # convert temporal derivative to column vector and multiply by -1 
    error_vec = -1 * img_error.flatten()

    # concatenate the gradients
    I_grad = np.column_stack((image_dir_x, image_dir_y))

    # solve the optical flow equation by finding the least square solution
    delta_p = np.linalg.lstsq(I_grad, error_vec)[0]
    
    return delta_p


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


def DisplayAll(img, box, I_crop, T_crop, dt, image_dir_x, image_dir_y, p, frame):
     # create figure
    fig = plt.figure()

    # setting values to rows and column variables
    rows, columns = 2, 3

    # Adds a subplot 
    fig.add_subplot(rows, columns, 1)
    plt.imshow(np.int16(I_crop), cmap='gray', vmin=-255, vmax=255)
    plt.axis('off')
    plt.title("I_crop")


     # Adds a subplot 
    fig.add_subplot(rows, columns, 2)
    plt.imshow(np.int16(T_crop), cmap='gray', vmin=-255, vmax=255)
    plt.axis('off')
    plt.title("T_crop")

    # Adds a subplot 
    fig.add_subplot(rows, columns, 3)
    plt.imshow(np.int16(dt), cmap='gray', vmin=-255, vmax=255)
    plt.axis('off')
    plt.title("dt")

    # Adds a subplot 
    fig.add_subplot(rows, columns, 4)
    plt.imshow(np.int16(image_dir_x), cmap='gray', vmin=-255, vmax=255)
    plt.axis('off')
    plt.title("image_dir_x")

    # Adds a subplot 
    fig.add_subplot(rows, columns, 5)
    plt.imshow(np.int16(image_dir_y), cmap='gray', vmin=-255, vmax=255)
    plt.axis('off')
    plt.title("image_dir_y")

    # Display the image
    fig.add_subplot(rows, columns, 6)
    plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)

    # Get the current reference
    ax = plt.gca()

    x_pos = box[0] + box[2]//2
    y_pos = box[1] + box[3]//2

    plt.quiver(x_pos, y_pos, -1*p[0], -1*p[1],
               color='b', scale=12)
            
    # Create a Rectangle patch
    rect = Rectangle((box[0], box[1]), box[2], box[3], 
                      linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    
    plt.title("Tracking")

    plt.savefig('output/{}.png'.format(frame))

    #plt.show()
    plt.clf()


def saveTracking(img, box, p, frame):
    """ saves frames to output folder """
    plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)

    # Get the current reference
    ax = plt.gca()

    x_pos = box[0] + box[2]//2
    y_pos = box[1] + box[3]//2

    plt.quiver(x_pos, y_pos, -1*p[0], -1*p[1],
               color='b', scale=200)
            
    # Create a Rectangle patch 
    rect = Rectangle((box[0], box[1]), box[2], box[3], 
                      linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    
    plt.title("Tracking")

    plt.savefig('output/{}.png'.format(frame))

    #plt.show()
    plt.clf()


def user_bounding_box(frame):
    # select ROI
    r = cv.selectROI("select the area", frame)
        
    return r


def GuassPyramidLayers(img, num=4, scale=2):
    """ applies gaussian subsampling to an image num times with reduction scale scale"""
    layers = [img]
    for _ in range(num):
        img = np.double(cv.pyrDown(img))
        layers.insert(0, img)

    skip = int(scale/2)
    layers[::skip]

    return layers 


def frame_scale(template, box, p, scale_factor, up=False, down=False):
    if up:
        template = template * scale_factor
        box = box * scale_factor
        p = p * scale_factor

    elif down:
        template = np.int32(np.floor(template / scale_factor))
        box = np.int32(np.floor(box / scale_factor))
        p = np.int32(np.floor(p / scale_factor))

    return template, box, p


def main():
    # read in frame 1
    base_frame = np.double(cv.imread("input/0.png", 0))

    # user picks region to track
    #template = np.array(user_bounding_box(base_frame))
    template = np.array((638, 236, 320, 391))
    box = np.array((638, 236, 320, 391))

    # translation vector 
    p = np.array((0, 0))

    # the number of layers -1 (not including the initial image layer)
    #user_pyramid =int(input("Number of Guass Pyramid levels: "))
    user_pyramid = 3
    scale_factor = 2 # must be power of 2 

    # pyramid for the inital image 
    T = GuassPyramidLayers(base_frame, user_pyramid, scale_factor)
    
    # scale to correct size
    template, box, p = frame_scale(template, box, p, 2**user_pyramid, down=True)

    # loop through frames 
    for i in range(350): 
        print(p)
        # load frames from input directory as grayscale 
        frame_new = np.double((cv.imread("input/{}.png".format(i), 0)))

        # pyramid for the image at time t
        I = GuassPyramidLayers(frame_new, user_pyramid, scale_factor)

        # compute p for the top level of the pyramid 
        delta_p = track_image(T[0], I[0], template, box, p)
        delta_p[0] = np.sign(delta_p[0])*(np.ceil(abs(delta_p[0])))
        delta_p[1] = np.sign(delta_p[1])*(np.ceil(abs(delta_p[1])))
        
        p = np.int32(p + delta_p)

        # update the tracking box cordinates 
        box = np.array((template[0] - p[0], template[1] - p[1], template[2], template[3]))

        # iterate through the rest of the pyramid levels
        for j in range(1, user_pyramid+1):
            template, box, p = frame_scale(template, box, p, scale_factor, up=True)
            
            # compute p for the jth level of the pyramid 
            delta_p = track_image(T[j], I[j], template, box, p)
            delta_p[0] = np.sign(delta_p[0])*(np.ceil(abs(delta_p[0])))
            delta_p[1] = np.sign(delta_p[1])*(np.ceil(abs(delta_p[1])))
            
            p = np.int32(p + delta_p)

            # update the tracking box cordinates 
            box = np.array((template[0] - p[0], template[1] - p[1], template[2], template[3]))

        template, box, p = frame_scale(template, box, p, 2**user_pyramid, down=True)

        # # calculate the image gradient w.r.t both x and y directions
        # image_dir_x, image_dir_y = image_gradient(I[0], Flatten=False)

        # # crop image frames, and convert to column vectors 
        # image_dir_x = image_dir_x[box[1]:box[3]+box[1], 
        #                         box[0]:box[2]+box[0]]
        
        # image_dir_y = image_dir_y[box[1]:box[3]+box[1], 
        #                         box[0]:box[2]+box[0]]

        # # crop image frames 
        # T_crop = T[0][template[1]:template[3]+template[1], 
        #           template[0]:template[2]+template[0]]
        
        # I_crop = I[0][box[1]:box[3]+box[1], 
        #           box[0]:box[2]+box[0]]
        
        # img_error, count = calculate_image_diff(T_crop, I_crop)
        
        # DisplayAll(I[0], box, I_crop, T_crop, img_error, image_dir_x, image_dir_y, p, frame=i)

        saveTracking(I[-1], box*2**3, p, frame=i)

        print(i)
    
    makeVideo()

if __name__ == "__main__":
    main()