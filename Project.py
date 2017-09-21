import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg
import pickle
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # n is window size of the moving average
        self.n = n
        # Polynimial coefficients: x = A*y**2 + B*y + C
        self.A = []
        self.B = []
        self.C = []
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.
    def get_fit(self):
        return (self.A_avg, self.B_avg, self.C_avg)
    def add_fit(self, fit):
        q_full = len(self.A)
        self.A.append(fit[0])
        self.B.append(fit[1])
        self.C.append(fit[2])

        if q_full >= self.n:
            self.A.pop(0)
            self.B.pop(0)
            self.C.pop(0)

        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)

        return (self.A_avg, self.B_avg, self.C_avg)

# SanityCheck:
#     # check that they have similar curvature
#     # check that they are separated by approx. the right distance horizontally
#     # check that they are roughly parallel

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

def DrawArea(undist, warped_size, src, dst, left_fit, right_fit):
    Minv = cv2.getPerspectiveTransform(dst, src)
    ploty = np.linspace(0, warped_size[0] - 1, warped_size[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros(warped_size).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped_size[1], warped_size[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

def Undistort(img, mtx, dist):
    # Assume input img is mpimg.imread (RGB)
    # transform to BGR to fit cv2.imread
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dst_img = cv2.undistort(img_BGR, mtx, dist, None, mtx)

    return cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

def warper(img, img_size, src, dst):
    # Compute and apply perpective transform
    img_size = tuple(img_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def draw_lines(points):
    # Transform float32 array into list
    points_t = list(map(list, points))
    lines = [[points_t[0] + points_t[1]], \
            [points_t[1] + points_t[2]], \
            [points_t[2] + points_t[3]], \
            [points_t[3] + points_t[0]]]
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot((x1, x2), (y1, y2), color='r', linewidth=2)

def SlidingWindow(img):
    # Assuming you have created a warped binary image called "binary_warped"
    binary_warped = img
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
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
    margin = 58
    # Set minimum number of pixels found to recenter window
    minpix = 65
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
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        #               (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
        #               (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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

    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds
    return ret

def SkipSlidingWindow(img, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    binary_warped = img
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If little relevant pixels found, return None, indicating ERROR
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret

def curvature(warped_size, left_fit, right_fit):
    ploty = np.linspace(0, warped_size[0] - 1, warped_size[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 600  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    ploty = np.linspace(0, warped_size[0] - 1, warped_size[0])
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    curv = np.mean((left_curverad, right_curverad))
    # print(left_curverad, 'm', right_curverad, 'm', curv, 'm')

    xleft_eval = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    xright_eval = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    xmean = np.mean((xleft_eval,xright_eval))
    offset = (warped_size[1]/2 - xmean) * xm_per_pix # +: car in right; -: car in left side
    return curv, offset

def process_image(image):
    # import Camera Calibration Parameters
    dist_pickle = "./wide_dist_pickle.p"
    with open(dist_pickle, mode="rb") as f:
        CalData = pickle.load(f)
        mtx, dist = CalData["mtx"], CalData["dist"]

    img_size = [image.shape[1], image.shape[0]]
    img_shape = (image.shape[0], image.shape[1])

    src = np.float32(
        [[(img_size[0] / 2) - 65, img_size[1] / 2 + 100],
         [((img_size[0] / 6) + 70), img_size[1] - 60],
         [(img_size[0] * 5 / 6) - 40, img_size[1] - 60],
         [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4) + 75, 0],
         [(img_size[0] / 4) + 75, img_size[1] - 60],
         [(img_size[0] * 3 / 4) - 5, img_size[1] - 60],
         [(img_size[0] * 3 / 4) - 5, 0]])

    undist_img = Undistort(image, mtx, dist)
    binary = pipeline(undist_img)
    warped = warper(binary, img_size, src, dst)
    leftline = Line()
    rightline = Line()

    left_fit, right_fit = SlidingWindow(warped)

    while(leftline.detected & rightline.detected):
        leftline.recent_xfitted.append(left_fit)
        rightline.recent_xfitted.append(right_fit)
        leftline.bestx
        leftline.best_fit = np.mean()

    if leftline.detected | rightline.detected:
        left_fit, right_fit = SlidingWindow(warped)
        leftline.detected, rightline.detected = True, True
        leftline.recent_xfitted.append(left_fit)
        rightline.recent_xfitted.append(right_fit)
        leftline.bestx
        leftline.best_fit = np.mean()
        
    else:
        left_fit, right_fit = SkipSlidingWindow(warped, left_fit, right_fit)

    line_curv, offset = curvature(img_shape, left_fit, right_fit)  # +: car in right; -: car in left side

    leftline = Line(left_fit, img_shape)




    result = DrawArea(undist_img, img_shape, src, dst, left_fit, right_fit)

    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    text1 = 'Radius of Curvature = %d(m)'
    text2 = 'Vehicle is %.2f(m) left of center'
    result_text = cv2.putText(result, text1 % (int(line_curv)),
                              (60, 100), font, 1.0, (255, 255, 255), thickness=2)
    result_text = cv2.putText(result, text2 % (-offset),
                              (60, 130), font, 1.0, (255, 255, 255), thickness=2)
    return result_text

# image = mpimg.imread('./test_images/test2.jpg')
# result = process_image(image, mtx, dist)
# plt.figure(figsize=(12,5))
# plt.imshow(result)


white_output = './output_videos/project_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below

input_path = './test_videos/project_video.mp4'
clip1 = VideoFileClip(input_path).subclip(5,8)
# clip1 = VideoFileClip(input_path)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)



#
# plt.figure(figsize=(12,5))
# plt.subplot(221)
# plt.imshow(undist_img)
# plt.subplot(222)
# plt.imshow(binary, cmap='gray')
# draw_lines(src)
# plt.subplot(223)
# plt.imshow(warped, cmap='gray')
# draw_lines(dst)
# plt.subplot(224)
# plt.imshow(result)