"""
Steps - :

Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.

"""

# import necessary modules
import cv2
import numpy as np
import scipy.misc as sci
import matplotlib.pyplot as plt

class Line():
  def __init__(self):
    #if line was deteced in last iteration
    self.curve = {'lane_info': ''}

# fit polynomial using sliding window for lane identification
def sliding_window_polyfit(binary_warped):
    # Take a histogram of the bottom quarter of the image
    histogram = np.sum(binary_warped[300:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 30
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 140, 0), 2)
        # print('rectangle 1', (win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 140, 0), 2)
        # print('rectangle 2', (win_xright_low,win_y_low), (win_xright_high,win_y_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # result visualization
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [30, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 30]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    return leftx, rightx, lefty, righty, ploty, left_fitx, right_fitx

# calculate radius of curvature
def rad_of_curv_calc(leftx, rightx, lefty, righty, ploty, left_fitx, right_fitx, lane_data):
    # convert from pixel space to meter space
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # calculate radisu of curvature
    left_eval = np.max(lefty)
    right_eval = np.max(righty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * left_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * right_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # calculate left_min by finding minimum value in first index of array
    left_min = np.amin(leftx, axis=0)
    # print('left_min', left_min)
    right_max = np.amax(rightx, axis=0)
    # print('right max', right_max)
    actual_center = (right_max + left_min) / 2
    dist_from_center = actual_center - (1280 / 2)
    # print('pix dist from center', dist_from_center)

    meters_from_center = xm_per_pix * dist_from_center
    string_meters = str(round(meters_from_center, 2))

    lane_info = 'left: ' + str(round(left_curverad, 2)) + ', right: ' + \
                str(round(right_curverad, 2)) + ', dist from center: ' + string_meters
    # print('Detected lane details: ', lane_info)

    # check if lane info exists
    if not lane_data.curve['lane_info'] or (abs(left_curverad - right_curverad) < 5000):
        lane_data.curve['lane_info'] = lane_info
        lane_data.curve['left_fitx'] = left_fitx
        lane_data.curve['lefty'] = lefty
        lane_data.curve['right_fitx'] = right_fitx
        lane_data.curve['righty'] = righty
        lane_data.curve['ploty'] = ploty
    else:
        # information from last iteration
        lane_info = lane_data.curve['lane_info']
        left_fitx = lane_data.curve['left_fitx']
        lefty = lane_data.curve['lefty']
        right_fitx = lane_data.curve['right_fitx']
        righty = lane_data.curve['righty']
        ploty = lane_data.curve['ploty']


    return left_fitx, lefty, right_fitx, righty, ploty, lane_info

# draw lanes on original image
def draw_lanes(img, warped, left_fitx, right_fitx, ploty, offset, win_bottom, win_top, win_height, y_bottom):
    # create img to draw the lines on
    no_warp = np.zeros_like(warped).astype(np.uint8)
    img_warp = np.dstack((no_warp, no_warp, no_warp))

    # recast x and y into usable format for cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # print('pts left', pts_left.shape, 'pts right', pts_right.shape)
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank img
    cv2.fillPoly(img_warp, np.int_([pts]), (0, 255, 0))

    # define source and destination
    img_size = (img.shape[1], img.shape[0])
    dst = np.float32([[img.shape[1] * (.5 - win_top / 2), img.shape[0] * win_height],
                      [img.shape[1] * (.5 + win_top / 2), img.shape[0] * win_height], \
                      [img.shape[1] * (.5 + win_bottom / 2), img.shape[0] * y_bottom],
                      [img.shape[1] * (.5 - win_bottom / 2), img.shape[0] * y_bottom]])
    src = np.float32(
        [[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

    # perspective transform
    Minv = cv2.getPerspectiveTransform(src, dst)
    # warp the blank back oto the original image using inverse perspective matrix
    newwarp = cv2.warpPerspective(img_warp, Minv, (img.shape[1], img.shape[0]))

    # combine the result with the original
    lane_marked_image = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return lane_marked_image

# detect lane lines & compute curvature
def detect_lane(binary_warped):
    # fit sliding window for lane detection
    leftx, rightx, lefty, righty, ploty, left_fitx, right_fitx = sliding_window_polyfit(binary_warped)

    # compute radius of curvature
    lane_data = Line()
    left_fitx, lefty, right_fitx, righty, ploty, lane_info = \
        rad_of_curv_calc(leftx, rightx, lefty, righty, ploty, left_fitx, right_fitx, lane_data)

    return left_fitx, lefty, right_fitx, righty, ploty, lane_info










#
#
#
#
# # Define a class to receive the characteristics of each line detection
# class Line():
#     def __init__(self):
#         # was the line detected in the last iteration?
#         self.detected = False
#         # x values of the last n fits of the line
#         self.recent_xfitted = []
#         #average x values of the fitted line over the last n iterations
#         self.bestx = None
#         #polynomial coefficients averaged over the last n iterations
#         self.best_fit = None
#         #polynomial coefficients for the most recent fit
#         self.current_fit = [np.array([False])]
#         #radius of curvature of the line in some units
#         self.radius_of_curvature = None
#         #distance in meters of vehicle center from the line
#         self.line_base_pos = None
#         #difference in fit coefficients between last and new fits
#         self.diffs = np.array([0,0,0], dtype='float')
#         #x values for detected line pixels
#         self.allx = None
#         #y values for detected line pixels
#         self.ally = None
#
