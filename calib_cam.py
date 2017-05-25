"""
Step - 1 & 2:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
"""

# import necessary modules
import numpy as np
import cv2
# import glob
import pickle

################### Calibrate camera ###################
#########################################################
def calibrate_camera(images):
    # array to store image points and object points from all the images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # prepare object points like (0,0,0),(1,0,0),(2,0,0),...,(8,5,0)
    nx = 9  # corners along horizontal
    ny = 6  # corners along vertical
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y coordinates

    # walk through images and find corners
    for image_ID, filename in enumerate(images):
        # print("calibrating for image: ", filename)
        # read image and convert to grayscale
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # if corners are found, add object points, image points
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # draw and save as new file
            img_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            new_filename = './output_images/camera_cal_files/corners_marked_' + filename[13:]
            cv2.imwrite(new_filename, img_with_corners)
            print("saved calibrated file: ", filename[13:])

    # 3D image size
    img_shape = (img.shape[1], img.shape[0])

    print("DONE: Reading images and identifying corners")

    return objpoints, imgpoints, img_shape

# save camera calibration matrix for later use
def save_calib_params(objpoints, imgpoints, img_shape):
    # Camera calibration, given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # save calibration matrix and distance
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./camera_calib_dist_pickle.p", "wb"))

################### undistort test images ###################
#########################################################
def undistort_images(test_images, calib_pickle_file):
    # read camera calibration parameters from pickle file
    with open(calib_pickle_file, 'rb') as pick:
        dist_pickle = pickle.load(pick)
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']

    # apply distortion correction on test images
    for image_ID, test_img_name in enumerate(test_images):
        print('undistorting image: ', test_img_name)

        img = cv2.imread('./test_images/' + test_img_name)
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        new_filename = './output_images/undistorted_test_images/undistorted_' + test_img_name
        cv2.imwrite(new_filename, undist_img)

# #########################################################
# ################### main function body ###################
# #########################################################
# # read all images in a folder with similar names
# images = glob.glob(('./camera_cal/calibration*.jpg'))
# test_images = glob.glob('./test_images/test*.jpg')
#
# # calibrate camera and read object-points (3D), image points (2D) and image shape
# objpoints, imgpoints, img_shape = calibrate_camera(images)
# print("DONE: Camera calibration")
#
# # save calibration parameters' pickle file
# save_calib_params(objpoints, imgpoints, img_shape)
# print("Calibration parameters pickle file saved \n")


