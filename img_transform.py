"""
Step - 3 & 4:

Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
"""

## import necessary modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

############### perspective transform #################
def perspective_transform(img, offset, win_bottom, win_top, win_height, y_bottom):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # print(img.shape)

    # # define window parameters
    # offset = 0.2 * img.shape[1] # offset for dst points
    # win_bottom = .76
    # win_top = .08
    # win_height = .62
    # y_bottom = .96

    # define source and destination
    src = np.float32([[img.shape[1] * (.5 - win_top / 2), img.shape[0] * win_height],
                      [img.shape[1] * (.5 + win_top / 2), img.shape[0] * win_height], \
                      [img.shape[1] * (.5 + win_bottom / 2), img.shape[0] * y_bottom],
                      [img.shape[1] * (.5 - win_bottom / 2), img.shape[0] * y_bottom]])
    dst = np.float32(
        [[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

    # print(src)
    # print(dst)

    # create transformation matrix based on the source and destination points
    M = cv2.getPerspectiveTransform(src, dst)
    # transform the image to birds eye view
    warped = cv2.warpPerspective(img, M, img_size)

    # plt.imshow(warped, cmap='gray')
    # plt.show()

    return warped

############### sobel threshold #################
# calculate threshold of x or y sobel
def abs_sobel_thresh(img, orient, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # binary_output = np.copy(img) # Remove this line
    return binary_output

############### magnitude threshold #################
# calculate magnitude of gradient for given image and threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    gradmag = np.uint8(gradmag / (np.max(gradmag) / 255))
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # binary_output = np.copy(img) # Remove this line
    return binary_output

############### direction threshold #################
# calculate gradient direction
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output

############### HLS channel selection ###############
# get S-Channel from HLS
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:, :, 2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # binary_output = np.copy(img) # placeholder line
    return binary_output

############### HSV channel selection ###############
# get V-Channel from HSV
def hsv_select(img, thresh=(0, 255)):
    # 1) Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 2) Apply a threshold to the V channel
    V = hsv[:, :, 2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(V)
    binary_output[(V > thresh[0]) & (V <= thresh[1])] = 1

    return binary_output

############### gray image thresholding ###############
def gray_threshold(img, thresh=(0,255)):
    # rgb = img
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    # thresh = (150, 255)

    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    return binary_output

############### R color thresholding ###############
def R_threshold(img, thresh=(0,255)):
    R = img[:,:,0]

    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1

    return binary_output

############### B color thresholding ###############
def B_threshold(img, thresh=(0,255)):
    B = img[:,:,2]

    binary_output = np.zeros_like(B)
    binary_output[(B > thresh[0]) & (B <= thresh[1])] = 1

    return binary_output

############# combination threshold ##################
def combined_thresh(img):

    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 120))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 90))
    mag_binary = mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 50))
    dir_binary = dir_threshold(img, sobel_kernel=9, thresh=(0.8, 1.2))
    hls_binary = hls_select(img, thresh=(110, 255))
    hsv_binary = hsv_select(img, thresh=(50, 255))
    gray_binary = gray_threshold(img, thresh=(50, 255))

    R_binary = R_threshold(img, thresh=(200, 255))
    B_binary = B_threshold(img, thresh=(210, 255))

    binary_output = np.zeros_like(dir_binary)

    # try combos
    # combo1 = ((mag_binary == 1) | (dir_binary == 1)) & (hls_binary == 1)
    # combo2 = (gradx == 1) & (hls_binary == 1)
    # combo3 = (gradx == 1) & (grady == 1) & (hls_binary == 1)
    combo4 = ((gradx == 1) & (grady == 1)) | ((hsv_binary == 1) & (hls_binary == 1))
    combo5 = combo4 & ((mag_binary == 1) | (dir_binary == 1))
    combo6 = combo4 & ((mag_binary == 1) & (gray_binary == 1))
    combo7 = combo4 & ((mag_binary == 1) & (dir_binary == 1) & (gray_binary == 1))
    combo8 = combo4  & ((R_binary == 1) & (dir_binary == 1))
    combo9 = combo4 | ((R_binary == 1) & ((mag_binary == 1) | (dir_binary == 1)))
    combo10 = combo4 | ((R_binary == 1) & (B_binary == 0))
    combo11 = combo4 & ((R_binary == 1) & (B_binary == 0) | (dir_binary == 1))


    final_combo = combo4 & (R_binary == 1)

    binary_output[final_combo] = 1

    return binary_output

# ################################################
# ########### main function #####################
# ################################################
# image_path = './output_images/undistorted_test_images/'
# image = mpimg.imread(image_path + 'undistorted_test4.jpg')
#
# thresh_img = combined_thresh(image)
#
# # plot to compare original and transformed images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(thresh_img, cmap='gray')
# ax2.set_title('Thresholded Combined', fontsize=20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()
