import cv2 as cv
import numpy as np
import numpy as np
import glob

def cameraMat():

    images = sorted(glob.glob('C:/Users/husse/Desktop/MSc. Robotcs/Sensing and Preciption/corseWork/calibration/*.png'))# This path will be changed on your PC; please change this directory to the one that contains the calibration images.
    img = cv.imread(images[0])# Read the first image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert to a gray scale image
    retval, corners = cv.findChessboardCorners(image=gray, patternSize=(9,6))# Gives the x and y position of the corners in the photo's matrix
    corners = np.squeeze(corners) # Get rid of extraneous singleton dimension
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Set termination criteria. This will stop after 30 iterations or when the error is less than 0.001
    corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=criteria) # extract refined corner coordinates.

    obj_grid = np.zeros((9*6,3), np.float32) 
    obj_grid[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # store the x and y positions of the corned in 3d space
    # Initialize enpty list to accumulate coordinates
    obj_points = [] # 3d world coordinates
    img_points = [] # 2d image coordinates
    

    for fname in images:
        print('Loading {}'.format(fname))
        img= cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        retval, corners = cv.findChessboardCorners(gray, (9,6))
        if retval:
            obj_points.append(obj_grid)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_points.append(corners2)

    retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return mtx

