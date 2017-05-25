"""
The goals / steps of this project are the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""

# import necessary modules
import os
import glob
from moviepy.editor import VideoFileClip
from calib_cam import *
from img_transform import *
from find_lane import *

############## compare images #####################
def compare_images(image, modified_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(modified_image, cmap='gray')
    ax2.set_title('modified image', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

############## camera calibration ##################
def cam_calibration():
    """
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    """
    # read all calibration images in a folder with similar names
    images = glob.glob('./camera_cal/calibration*.jpg')

    # calibrate camera and read object-points (3D), image points (2D) and image shape
    objpoints, imgpoints, img_shape = calibrate_camera(images)
    print("DONE: Camera calibration")
    # save calibration parameters' pickle file
    save_calib_params(objpoints, imgpoints, img_shape)
    print("Calibration parameters pickle file saved ")

############## distortion correction ##################
def image_undistort():
    """
    Apply a distortion correction to raw test images.
    """
    # read test images
    all_test_images = os.listdir('test_images')
    test_images = []
    for name in all_test_images:
        if name.endswith(".jpg"):
            test_images.append(name)
    # apply distortion correction on test images
    undistort_images(test_images, './camera_calib_dist_pickle.p')
    print("DONE: undistorted test-images saved")

###################################################
# main function for image processing
def process_image(img):

    # undistort image (needed for video editing)
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    # apply thresholding
    thresh_img = combined_thresh(undist_img)
    compare_images(image, thresh_img)

    # define window parameters
    offset = 0.20 * img.shape[1]  # offset for dst points
    win_bottom = 0.76 #.76
    win_top = .08
    win_height = .62
    y_bottom = .94

    # apply perspective transform to region of interest
    warped_img = perspective_transform(thresh_img, offset, win_bottom, win_top, win_height, y_bottom)
    # compare_images(image, warped_img)

    # detect lane and compute curvature
    left_fitx, lefty, right_fitx, righty, ploty, lane_info = detect_lane(warped_img)

    # draw lane on image
    lane_marked_image = draw_lanes(img, warped_img, left_fitx, right_fitx, ploty, offset, win_bottom, win_top, win_height, y_bottom)
    plt.imshow(lane_marked_image)
    plt.show()

    return lane_marked_image

#####################################################################
###################### main function body ###########################
#####################################################################
# calibrate camera & save undistorted images
# cam_calibration() # comment after successful run
# image_undistort() # comment after successful run

# load undistortion matrix from camera
with open('./camera_calib_dist_pickle.p', 'rb') as pick:
    dist_pickle = pickle.load(pick)

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

testcase = 'image' # 'image OR 'video' OR frames
################################
if testcase=='image':
    ## section for single image test
    # # load test image
    image_path = './output_images/undistorted_test_images/'
    image = mpimg.imread(image_path + 'undistorted_test4.jpg')

    process_image(image)

elif testcase=='frame':
    ## section for single image test
    # # load test image
    # image_path = './video_frames/'
    # for frame_num in range(620, 800):
    #     print("frame: ", frame_num)
    #     image = mpimg.imread(image_path + 'frame' + str(frame_num) + '.jpg')
    #     process_image(image)
    #     # bad frames: 664, 688, 689, 690, 691, 700, 701, 712, 713, 714, 715

    frame_num = 713 # 1043 is ok
    image_path = './video_frames/'
    image = mpimg.imread(image_path + 'frame' + str(frame_num) + '.jpg')

    process_image(image)

elif testcase=='video':
    print("Starting video feed")
    proj_output = "./output_video_temp.mp4"
    clip1 = VideoFileClip("./project_video.mp4")
    output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    output_clip.write_videofile(proj_output, audio=False)

    # challenge_output = "output_challenge_video.mp4"
    # clip1 = VideoFileClip("challenge_video.mp4")
    # output_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    # output_clip.write_videofile(challenge_output, audio=False)



