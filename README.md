#**Self-Driving Car Engineer Nanodegree** 

##Project-3: Advanced Lane Finding

###Writeup author: Vikalp Mishra

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Orignal chessboard image"
[image2]: ./output_images/camera_cal_files/corners_marked_calibration2.jpg
[image3]: ./test_images/test4.jpg "test-image-4"
[image4]: ./output_images/undistorted_test_images/undistorted_test4.jpg "undistorted test-image-4"
[image5]: ./output_images/test4_combo_threshold.jpg "binary thresholding of test-image-4"
[image6]: ./output_images/test4_perspective.jpg "Birds-eye-view, test-image-4"
[image7]: ./output_images/tes4_sliding_window.jpg "lane line identification"
[image8]: ./output_images/test4_lane_overlap.jpg "best fit lanes overlapped on to the original image"
[image9]: ./camera_cal/calibration1.jpg "original camera-cal image-1"
[image10]: ./output_images/undist_calibration1.jpg "undistorted camera-cal image-1"


## Overview ##
For the given video of a car driving in its lane on a road, the project tries to identify the lane and highlight it in each frame of the video. The results are shown in video "output.mp4"

Steps I have taken to achieve this are similar to that outlined above in the goals, namely:

1. Camera calibration
2. Undistort images
3. Create thresholded binary images
4. Apply perspective transform to get birds-eye-view
5. Detect lane pixels and compute lane curvature
6. warp detected lane onto the original image

`adv_lane_finding.py` is the main script used for all steps listed above, with appropriate sections commented as necessary.

---

### 1. Camera Calibration

Camera calibration is performed using the 20 chessboard images provided in the `camera_cal` folder. I use opencv function `drawChessboardCorners` within a custom function `calibrate_camera()` to look for 9x6 inner corners in each of the 20 image that have been taken from different angles and distances. 

A sample original chessboard image and image with marked inner cornes is shown in [image1] and [image2]

![alt text][image1]
![alt text][image2]

Once the inner corners are identified, the 3D `objpoints` and 2D `imgpoints` (with z=0) lists are appended and passed to opencv function `calibrateCamera()` which returns the calibration and distortion matrix that are saved using python pickle structure `camera_calib_dist_pickle.p` for later use. The script is similar to that developed in one of the lectures and is part of `calib_cam.py`

### 2. Pipeline (single image)
Here I describe the pipeline used for lane detection in a single image, that is later used on each frame in the output video.

#### 2.1 Distortion correction
Given the camera calibration and distortion matrix, distortion correction is applied to each image using the opencv function `undistort()`. The routine is part of `calib_cam.py` and the function is named `undistort_images()`

A sample test image is shown here in [image3]

![alt text][image3]

Notice the change in location of white car after the image is undistorted, as shown in [image4]

![alt text][image4]

The distortion correction using a chessboard image is highlighted below, where the top image is the original image with distortion, ...

![alt text][image9]
... while the bottom image is the one after application of distortion correction (`calibration1.jpg` is the image used here).

![alt text][image10]

#### 2.2 Apply binary thresholds for lane line identification

I tried various methods and their combinations for applying binary thresholds to detect the lane lines. all these functions are part of script `img_transform.py`:

1. `abs_sobel_thresh()`: applies Sobel gradient thresholding along x and y direction of the image with a kernel size of 3.
2. `mag_thresh()`: creates a magnitude threshold based on the combined values of the sobel in x and y directions.
3. `dir_threshold()`: sets a orientation threshold for given bounds on angle (default being between 0 and pi/2).
4. `hls_select()`: applies binary thresholding using saturation channel of HLS color spectrum
5. `hsv_select()`: applies binary thresholding using value channel of HSV color spectrum
6. `gray_threshold()`: applies binary thresholding using gray-scale image
7. `R_threshold()`: applies binary thresholding using R-color channel
8. `B_threshold()`: applies binary thresholding using B-color channel
9. `combined_thresh()`: uses various combinations of the methods listed above for thresholding.

Parameters for each method were tuned and tested on provided `test_images` and finally a combination of (((x and y) or (hls and hsv)) and R-channel) was found to be best suited as it maximized lane pixel intensity while minimizing noise from other objects in the image, as is shown in [image5]

![alt text][image5]

#### 2.3 Perspective transform to birds-eye-view

The perspective transform function `perspective_transform()` is also part of the script `img_transform.py`. In this function, the road image is taken and mapped to a rectangular birds-eye-view region.

The source points (`src`) are cropped version of orginal image and proportional to the size of the image, and were determined by trial-and-error on all test images. The final selection was as follows:

```python
offset = 0.2 * img_size[0] # offset for dst points
    win_bottom = .76
    win_top = .08
    win_height = .62
    y_bottom = .96

    # define source and destination
    src = np.float32([[img.shape[1] * (.5 - win_top / 2), img.shape[0] * win_height],
                      [img.shape[1] * (.5 + win_top / 2), img.shape[0] * win_height], \
                      [img.shape[1] * (.5 + win_bottom / 2), img.shape[0] * y_bottom],
                      [img.shape[1] * (.5 - win_bottom / 2), img.shape[0] * y_bottom]])
    dst = np.float32(
        [[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

```

This resulted in the following source and destination points for test-image-4:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 446      | 256, 0        | 
| 691, 446      | 1024, 0      |
| 1126, 691     | 1024, 720      |
| 154, 691      | 256, 720       |

The criteria behind selecting these parameters was that in birds-eye-view the left and right lane should appear more or less parallel. A sample for test image-4 is shown below in [image6]

![alt text][image6]

#### 2.4 Detect left/right lane pixels and find curvature

The main script for finding lanes and calculating curvature is part of function `find_lane.py`. Within the python file, `detect_lane` is the main function which calls `sliding_window_polyfit` to slide a window over region of interest to detect the left and right lane pixels. This function is basically copied from several sections in udacity lectures.

Once potential lane-pixels are identified the curvate is computed using 2nd order polynomial suggestion in the lecture. A check is in place to make sure that the radial distance between lanes is less than 5000 meters. A sample output for test-image-4 is shown in [image7] where lanes are marked with yellow line.

![alt text][image7]

#### 2.5 Warp lanes on original image

Function named `draw_lane`, which is part of module `find_lane.py` overlaps lane in green on top of the original image for display. An inverse perspective transform is used for this purpose where source (`src`) and destination (`dst`) values are switched and passed to the opencv function `getPerspectiveTransform()`. The result looks like that shown in [image8]

![alt text][image8]

---
### 3. Video

The image pipeline described above is passed as a process to the `VideoFileClip` module imported from `moviepy.editor` similar to what was done for project-1 on lane-finding.

The video file is saved in [output_video.mp4](./output_video.mp4) as well as [output\_video\_with_text.mp4](./output_video_with_text.mp4)

---

### Discussion

The approach I have taken overall is described at appropriate places in this document above. Two relevant points to mention here, I think, are firstly, to identify sliding window start point, I am looking only at bottom quarter of the image and not bottom half as was done in the lecture notes. This is because after some playing around with the images I found that if I look at the bottom half, there is still a lot of noise in several cases depending on the angle of camera w.r.t. the normal to the road......or in case of computer vision terminology, I am just trying to not look past the potential `horizon` in an image.

Secondly, the lane detection is a little jittary in the presence of shadow effects, close-by artifacts (where by artifact I mean anything other than the lane itself, like presence of another car) and/or in case of faint lane markings.

To counter some of these issues it, storing history information and using that to predict lane location can be useful. What I mean is that I can potentially use lane marking information from last say 5 frames to better approximate the lane location and slope in the current frame, since the lane markings should remain continous curves.