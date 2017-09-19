'''
The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. - Done
2. Apply a distortion correction to raw images. - Done
3. Use color transforms, gradients, etc., to create a thresholded binary image. - Done
4. Apply a perspective transform to rectify binary image ("birds-eye view"). - Done
5. Detect lane pixels and fit to find the lane boundary. - Done
6. Determine the curvature of the lane and vehicle position with respect to center. - Done
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg

image = mpimg.imread('./test_images/test5.jpg')
img_size = [image.shape[1], image.shape[0]]
# ##############################
# Perspective Transformation   #
# ##############################
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def draw_lines(points):
    points_t = list(map(list, points))
    lines = [[points_t[0] + points_t[1]], \
            [points_t[1] + points_t[2]], \
            [points_t[2] + points_t[3]], \
            [points_t[3] + points_t[0]]]
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot((x1, x2), (y1, y2), color='r', linewidth=2)

'''
# ####################
# Camera Calibration #
# ####################
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/cal*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # write_name = './camera_cal/corners_found'+str(idx)+'.jpg'
        # cv2.imwrite(write_name, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()
'''
'''
# Undistortion
# Test undistortion on an image
img = cv2.imread('./test_images/test4.jpg') #cv2 read bgr channel
img_size = (img.shape[1], img.shape[0])
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./test_images/undist_t4.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
# pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# Visualize undistortion
# Convert bgr (cv2) to rgb (matplotlib)
img_bgr = img
img_rgb = np.zeros(img_bgr.shape, img_bgr.dtype)
img_rgb[:,:,0] = img_bgr[:,:,2]
img_rgb[:,:,1] = img_bgr[:,:,1]
img_rgb[:,:,2] = img_bgr[:,:,0]

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=30)
ax1.grid()
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
ax2.grid()
plt.show()
'''

# ####################
# Color & Gradient   #
# ####################
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Note: img is the undistorted image
    # img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
     # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

# #####################################
# Sliding Window Search - Convolution #
# #####################################
def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

binary = pipeline(image)
src = np.float32(
    [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
    [((img_size[0] / 6) + 70), img_size[1] - 60],
    [(img_size[0] * 5 / 6) - 40, img_size[1] - 60],
    [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4) + 75, 0],
     [(img_size[0] / 4) + 75, img_size[1] - 60],
     [(img_size[0] * 3/4) - 5, img_size[1] - 60],
     [(img_size[0] * 3/4) - 5, 0]])
warped = warper(binary, src, dst)

# window settings
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 30  # How much to slide left and right for searching

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    zero_channel = np.zeros_like(template)  # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    warpage = np.array(cv2.merge((warped*255, warped*255, warped*255)),
                       np.uint8)  # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(warped)
# ax2.set_title('Warped image with dest. points drawn', fontsize=20)

# Plot Perspective_Binary
# plt.figure(figsize=(12,5))
# plt.subplot(121)
# plt.imshow(binary, cmap='gray')
# draw_lines(src)
# plt.title('Threshold Image', fontsize=20)
# plt.subplot(122)
# plt.imshow(warped, cmap='gray')
# draw_lines(dst)
# plt.title('Warped image with dest. points drawn', fontsize=20)

# Plot historgram
# plt.figure(figsize=(9,4))
# histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
# plt.plot(histogram)
# plt.xlabel('Pixel Position')
# plt.ylabel('Counts')





# # Assuming you have created a warped binary image called "binary_warped"
# binary_warped = warped
# # Assuming you have created a warped binary image called "binary_warped"
# # Take a histogram of the bottom half of the image
# histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
# # Create an output image to draw on and  visualize the result
# out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# # Find the peak of the left and right halves of the histogram
# # These will be the starting point for the left and right lines
# midpoint = np.int(histogram.shape[0]/2)
# leftx_base = np.argmax(histogram[:midpoint])
# rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#
# # Choose the number of sliding windows
# nwindows = 9
# # Set height of windows
# window_height = np.int(binary_warped.shape[0]/nwindows)
# # Identify the x and y positions of all nonzero pixels in the image
# nonzero = binary_warped.nonzero()
# nonzeroy = np.array(nonzero[0])
# nonzerox = np.array(nonzero[1])
# # Current positions to be updated for each window
# leftx_current = leftx_base
# rightx_current = rightx_base
# # Set the width of the windows +/- margin
# margin = 58
# # Set minimum number of pixels found to recenter window
# minpix = 65
# # Create empty lists to receive left and right lane pixel indices
# left_lane_inds = []
# right_lane_inds = []
#
# # Step through the windows one by one
# for window in range(nwindows):
#     # Identify window boundaries in x and y (and right and left)
#     win_y_low = binary_warped.shape[0] - (window+1)*window_height
#     win_y_high = binary_warped.shape[0] - window*window_height
#     win_xleft_low = leftx_current - margin
#     win_xleft_high = leftx_current + margin
#     win_xright_low = rightx_current - margin
#     win_xright_high = rightx_current + margin
#     # Draw the windows on the visualization image
#     cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
#     (0,255,0), 2)
#     cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
#     (0,255,0), 2)
#     # Identify the nonzero pixels in x and y within the window
#     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#     (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
#     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#     (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
#     # Append these indices to the lists
#     left_lane_inds.append(good_left_inds)
#     right_lane_inds.append(good_right_inds)
#     # If you found > minpix pixels, recenter next window on their mean position
#     if len(good_left_inds) > minpix:
#         leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
#     if len(good_right_inds) > minpix:
#         rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#
# # Concatenate the arrays of indices
# left_lane_inds = np.concatenate(left_lane_inds)
# right_lane_inds = np.concatenate(right_lane_inds)
#
# # Extract left and right line pixel positions
# leftx = nonzerox[left_lane_inds]
# lefty = nonzeroy[left_lane_inds]
# rightx = nonzerox[right_lane_inds]
# righty = nonzeroy[right_lane_inds]
#
# # Fit a second order polynomial to each
# left_fit = np.polyfit(lefty, leftx, 2)
# right_fit = np.polyfit(righty, rightx, 2)






# Visualization Finding the lines
# # Generate x and y values for plotting
# ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
# out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
# out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.show()
# # Define conversions in x and y from pixels space to meters
# ym_per_pix = 30/720 # meters per pixel in y dimension
# xm_per_pix = 3.7/700 # meters per pixel in x dimension
# y_eval = np.max(ploty)
# # Fit new polynomials to x,y in world space
# left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
# right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
# # Calculate the new radii of curvature
# left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
# right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# # Now our radius of curvature is in meters
# print(left_curverad, 'm', right_curverad, 'm')
# # Example values: 632.1 m    626.2 m


# Plot Sliding Window Search - Convoluntion
plt.imshow(output)
plt.title('window fitting results')
plt.show()