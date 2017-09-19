'''
The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. - Done
2. Apply a distortion correction to raw images. - Done
3. Use color transforms, gradients, etc., to create a thresholded binary image. - Done
4. Apply a perspective transform to rectify binary image ("birds-eye view"). - Done
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
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

image = mpimg.imread('./test_images/straight_lines2.jpg')

# ##############################
# Perspective Transformation   #
# ##############################
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def draw_lines(img, points, color=(255,0,0), thickness=2):
    points_t = list(map(list, points))
    lines = [[points_t[0] + points_t[1]], \
            [points_t[1] + points_t[2]], \
            [points_t[2] + points_t[3]], \
            [points_t[3] + points_t[0]]]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

img_size = [image.shape[1], image.shape[0]]

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

# src_test5 = np.float32(
#     [[(img_size[0] / 2) - 70, img_size[1] / 2 + 100],
#     [((img_size[0] / 6) + 50), img_size[1] - 60],
#     [(img_size[0] * 5 / 6) - 30, img_size[1] - 60],
#     [(img_size[0] / 2 + 75), img_size[1] / 2 + 100]])
# dst_test5 = np.float32(
#     [[(img_size[0] / 4) + 60, 0],
#      [(img_size[0] / 4) + 60, img_size[1] - 60],
#      [(img_size[0] * 3/4) + 5, img_size[1] - 60],
#      [(img_size[0] * 3/4) + 5, 0]])
warped = warper(image, src, dst)
draw_lines(image, src)
draw_lines(warped, dst)
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

# result = pipeline(image)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(warped)
ax2.set_title('Warped image with dest. points drawn', fontsize=20)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
