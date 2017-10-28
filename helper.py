import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg
from skimage import exposure


def warp(img):
    """
    Perspective Transformation 
    :param img: 
    :return: warped image
    """
    # Compute and apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1280, 720), flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def draw_lines(points):
    points_t = list(map(list, points))
    lines = [[points_t[0] + points_t[1]],
            [points_t[1] + points_t[2]],
            [points_t[2] + points_t[3]],
            [points_t[3] + points_t[0]]]
    for line in lines:
        for x1, y1, x2, y2 in line:
            plt.plot((x1, x2), (y1, y2), color='r', linewidth=2)


def undistort(img, mtx, dist):
    """
    Use cv2.undistort to undistort
    :param img: Assuming input img is RGB (imread by mpimg)
    :param mtx: camera calibration parameter
    :param dist: camera calibration parameter
    :return: Undistorted img
    """
    # transform to BGR to fit cv2.imread
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dst_img = cv2.undistort(img_BGR, mtx, dist, None, mtx)

    return cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)


def img2binary(img, s_thresh=(100, 255), sx_thresh=(20, 100)):
    """
    Color and Gradient
    :param img: undistorted image (RGB)
    :param s_thresh: 
    :param sx_thresh: 
    :return: combined_binary img
    """
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gaussian Blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)  # kernel size = 5
    blur = gray
    # Sobel x
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)  # Take the derivative in x
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    abs_bin = np.zeros_like(scaled_sobel)
    abs_bin[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Calculate the gradient magnitude
    mag_thresh = (50, 255)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_bin = np.zeros_like(gradmag)
    mag_bin[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Calculate the x and y gradients
    dir_thresh = (0, np.pi / 2)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_bin = np.zeros_like(absgraddir)
    dir_bin[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    # Threshold color channel
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    schannel_bin = np.zeros_like(s_channel)
    schannel_bin[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # green - gradient thres; blue - color channel thres
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack((np.zeros_like(abs_bin), abs_bin, schannel_bin)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(abs_bin)
    combined_binary[(schannel_bin == 1) | (((mag_bin == 1) & (dir_bin == 1)) | abs_bin == 1)] = 1

    return combined_binary


def sobelx_filter(img, sx_thresh=(20, 100)):
    warped = warp(img)
    # Grayscale image
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    abs_bin = np.zeros_like(scaled_sobel)
    abs_bin[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    return abs_bin


def luv_lab_filter(img, l_thresh=(210, 255), b_thresh=(140, 200), s_thresh=(90, 255), plot=False):
    # warped = warp(img)
    warped = img

    l = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)[:, :, 0]
    l_bin = np.zeros_like(l)
    l_bin[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1

    b = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)[:, :, 2]
    b_bin = np.zeros_like(b)
    b_bin[(b >= b_thresh[0]) & (b <= b_thresh[1])] = 1

    s = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)[:, :, 2]
    s_bin = np.zeros_like(s)
    s_bin[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1

    combine = np.zeros_like(l)
    # combine[(l_bin == 1) | ((b_bin == 1) & (s_bin == 1))] = 1
    combine[(l_bin == 1) | (b_bin == 1)] = 1
    if plot is True:
        plt.figure(figsize=(10, 6))
        plt.subplot(231)
        plt.imshow(warped)
        plt.subplot(232)
        plt.imshow(l_bin, cmap='gray')
        plt.title('L channel')
        plt.subplot(233)
        plt.imshow(b_bin, cmap='gray')
        plt.title('B channel')
        plt.subplot(234)
        plt.imshow(s_bin, cmap='gray')
        plt.title('S channel')
        plt.subplot(235)
        plt.imshow(combine, cmap='gray')
        plt.title('Combination')
        plt.subplot(236)
        plt.imshow(img)

        plt.show()

    return combine


# for convolution search method
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


# for convolution search method
def find_window_centroids(warped, window_width, window_height, margin):
    """
    convolution method to search "hot" pixel in each window
    :param warped: 
    :param window_width: 
    :param window_height: 
    :param margin: 
    :return: 
    """
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


def sliding_window(binary_warped):
    """
    Sliding window search for left and right lane.
    :param binary_warped: bird-eye binary image
    :return: ret dictionay including fit 
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[hist_top_boundary:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[hist_left_start:midpoint]) + hist_left_start
    rightx_base = np.argmax(histogram[hist_right_start:-200]) + hist_right_start

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
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 40
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
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

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

    # If little relevant pixels found, return None, indicating ERROR
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # for debug
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ret = {'left_fit': left_fit,
           'right_fit': right_fit,
           'left_fitx': left_fitx,
           'right_fitx': right_fitx,
           'nonzerox': nonzerox,
           'nonzeroy': nonzeroy,
           'out_img': out_img,
           'left_lane_inds': left_lane_inds,
           'right_lane_inds': right_lane_inds}

    return ret


def skip_sliding_window(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!

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

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ret = {'left_fit': left_fit,
           'right_fit': right_fit,
           'left_fitx': left_fitx,
           'right_fitx': right_fitx,
           'nonzerox': nonzerox,
           'nonzeroy': nonzeroy,
           'left_lane_inds': left_lane_inds,
           'right_lane_inds': right_lane_inds}

    return ret


def refine(undistort_color, left_fit_last, right_fit_last, skip=False):
    warped_binary = luv_lab_filter(undistort_color)
    ret = sliding_window(warped_binary) if skip is False \
        else skip_sliding_window(warped_binary, left_fit_last, right_fit_last)

    if ret is None:
        return None

    else:
        if (len(ret["left_lane_inds"]) < 500) | (len(ret["right_lane_inds"]) < 500):
            lighter = gamma_reset(undistort_color, gamma=0.4)
            luv_bin = luv_lab_filter(lighter, l_thresh=(220, 255), b_thresh=(140, 200))
            sobel_bin = sobelx_filter(lighter, sx_thresh=(20, 100))
            comb_bin = np.zeros_like(luv_bin)
            comb_bin[(luv_bin == 1) | (sobel_bin == 1)] = 1
            ret = sliding_window(comb_bin) if skip is False \
                else skip_sliding_window(warped_binary, left_fit_last, right_fit_last)
        if len(ret["nonzerox"]) > 90000:
            darker = gamma_reset(undistort_color, gamma=4)
            warped_binary = luv_lab_filter(darker)
            ret = sliding_window(warped_binary) if skip is False \
                else skip_sliding_window(warped_binary, left_fit_last, right_fit_last)
        return ret


def refine_equidistant(undistort_color, left_fit_last, right_fit_last, skip=False):
    """
    
    :param img: undistorted color img
    :param left_fit_last: 
    :param right_fit_last: 
    :param skip: 
    :return: ret_new dictionary
    """
    binary = combine_bin(undistort_color)
    binary_warped = warp(binary)
    if skip is False:  # Sliding_window search
        ret = sliding_window(binary_warped)
    else:  # Skip sliding_window search
        ret = skip_sliding_window(binary_warped, left_fit_last, right_fit_last)

    if ret is None:
        brighter = gamma_reset(undistort_color, gamma=0.3)
        binary = img2binary(brighter, s_thresh=(60, 255), sx_thresh=(20, 100))
        binary_warped = warp(binary)
        if skip is False:
            ret = sliding_window(binary_warped)
        else:
            ret = skip_sliding_window(binary_warped, left_fit_last, right_fit_last)

    left_fit, right_fit = ret["left_fit"], ret["right_fit"]
    left_fitx, right_fitx = ret["left_fitx"], ret["right_fitx"]
    left_lane_inds, right_lane_inds = ret["left_lane_inds"], ret["right_lane_inds"]
    left_pix_num, right_pix_num = len(left_lane_inds), len(right_lane_inds)
    nonzerox = ret["nonzerox"]
    # confidence left >= right or mis-detect non-lane in right
    if ((left_pix_num / right_pix_num >= 4) & (right_pix_num < 1000)) or \
            (right_pix_num >= 60000):
        lane_width = right_fitx[-100] - left_fitx[-100]
        right_fit = np.array(equidistant(left_fit, lane_width, max_l=500))
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # if (min(left_pix_num, right_pix_num) < 2000):

    ret_new = {'left_fit': left_fit,
                'right_fit': right_fit,
                'left_fitx': left_fitx,
                'right_fitx': right_fitx,
                'left_lane_inds': left_lane_inds,
                'right_lane_inds': right_lane_inds,
                'nonzerox': nonzerox}

    return ret_new


def visualize():
    """Plot the images for README"""
    # Plot the Perspective Transformation
    image = mpimg.imread('./test_images/straight_lines2.jpg')
    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.imshow(image)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    draw_lines(src)
    plt.title('Original Image', fontsize=12)
    plt.subplot(122)
    plt.imshow(warp(image))
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    draw_lines(dst)
    plt.title('Warped image with dest. points', fontsize=12)
    plt.show()

    # Plot Perspective_Binary
    # plt.figure(figsize=(8,2))
    # plt.subplot(121)
    # plt.imshow(binary, cmap='gray')
    # draw_lines(src)
    # plt.title('Threshold Image', fontsize=12)
    # plt.subplot(122)
    # plt.imshow(warped, cmap='gray')
    # draw_lines(dst)
    # plt.title('Warped image with dest. points drawn', fontsize=12)

    # Plot historgram
    # plt.figure(figsize=(9,4))
    # histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
    # plt.plot(histogram)
    # plt.xlabel('Pixel Position')
    # plt.ylabel('Counts')


def color_filter(img, r_th=120, g_th=100, b_th=50):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    bin_color = np.zeros_like(R)
    bin_color[(R >= r_th) & (G >= g_th) & (B >= b_th)] = 1

    return bin_color


def combine_bin(img, r_th=140, g_th=100, b_th=50,
                s_thresh=(90, 255), sx_thresh=(20, 100)):
    bin_thres = img2binary(img, s_thresh, sx_thresh)
    bin_color = color_filter(img, r_th, g_th, b_th)

    com_bin = np.zeros_like(bin_color)
    com_bin[(bin_thres == 1) & (bin_color == 1)] = 1

    return com_bin


def pol_calc(pol, x):
    pol_fit = np.poly1d(pol)  # least square method polynominal fitting (2 orders)
    return pol_fit(x)  # interpolation according to x


def equidistant(pol, distance, max_l=1, plot=False):
    y_pol = np.linspace(0, max_l, num=EQUID_POINTS)
    x_pol = pol_calc(pol, y_pol)
    y_pol *= IMAGE_H  # Convert y coordinates to [0...IMAHE_H] scale
    x_m = []
    y_m = []
    k_m = []
    for i in range(len(x_pol)-1):
        x_m.append((x_pol[i+1]-x_pol[i])/2.0+x_pol[i])  # Calculate points position between given points
        y_m.append((y_pol[i+1]-y_pol[i])/2.0+y_pol[i])
        if x_pol[i+1] == x_pol[i]:
            k_m.append(1e8)  # A vary big number
        else:
            k_m.append(-(y_pol[i+1]-y_pol[i])/(x_pol[i+1]-x_pol[i]))  # Slope of perpendicular lines
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    k_m = np.array(k_m)
    # Calculate equidistant points
    y_eq = distance * np.sqrt(1.0/(1+k_m**2))
    x_eq = np.zeros_like(y_eq)
    if distance >= 0:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    else:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    y_eq /= IMAGE_H # Convert all y coordinates back to [0..1] scale
    y_pol /= IMAGE_H
    y_m /= IMAGE_H
    pol_eq = np.polyfit(y_eq, x_eq, len(pol)-1)  # Fit equidistant with a polinomial
    if plot:  # Visualize results
        plt.figure()
        plt.plot(x_pol, y_pol, color='green', linewidth=1, label='Original line')  # Original line
        plt.plot(x_eq, y_eq, color='red', linewidth=3, label='Equidistant')  # Equidistant
        plt.plot(pol_calc(pol_eq, y_pol), y_pol, color='blue',
                 linewidth=1, label='Approximation')  # Approximation
        plt.legend()
        for i in range(len(x_m)):
            plt.plot([x_m[i],x_eq[i]], [y_m[i], y_eq[i]], color='black', linewidth=1)  # Draw connection lines

    return pol_eq


def gamma_reset(raw_img, gamma=0.3):
    """
    adjust gamma to deal with images too dark or too bright
    :param raw_img: 
    :param gamma: <1, to brighten; >1, to darken.
    :return: 
    """
    return exposure.adjust_gamma(raw_img, gamma)


def debug_threshold_all():
    # Test on images
    i = 1
    f, axes = plt.subplots(4, 6, figsize=(11, 7))
    f.tight_layout()
    axes_stack = np.hstack(axes)

    for image in glob.glob('./test_images/test_ch*.jpg'):
        img = mpimg.imread(image)

        img_1 = img2binary(img)
        img_2 = color_filter(img)
        img_3 = combine_bin(img)

        axes_stack[i-1].imshow(img)
        axes_stack[i - 1].axis('off')

        axes_stack[i + 5].imshow(img_1, cmap='gray')
        axes_stack[i + 5].axis('off')

        axes_stack[i + 11].imshow(img_2, cmap='gray')
        axes_stack[i + 11].axis('off')

        axes_stack[i + 17].imshow(img_3, cmap='gray')
        axes_stack[i + 17].axis('off')

        plt.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.45)
        i += 1


def debug_thres_single(img, r_th=140, g_th=100, b_th=50,
                s_thresh=(80, 255), sx_thresh=(15, 110)):
    # (3,1) figure

    plt.figure(figsize=(8, 11))

    img1 = img2binary(img, s_thresh, sx_thresh)
    img2 = color_filter(img, r_th, g_th, b_th)
    img3 = combine_bin(img, r_th, g_th, b_th, s_thresh, sx_thresh)

    plt.subplot(4,1,1)
    plt.imshow(img)
    plt.title('Raw Image')
    plt.axis('off')

    plt.subplot(4,1,2)
    plt.imshow(img1, cmap='gray')
    plt.title('Gradient & S Channel Binary')
    plt.axis('off')

    plt.subplot(4,1,3)
    plt.imshow(img2, cmap='gray')
    plt.title('Yellow & White Filter')
    plt.axis('off')

    plt.subplot(4,1,4)
    plt.imshow(img3, cmap='gray')
    plt.title('Combine Binary')
    plt.axis('off')

    plt.show()


def debug_pipeline(img):
    # input img is raw image
    # import Camera Calibration Parameters

    undist_img = undistort(img, mtx, dist)
    warped_color = warp(undist_img)
    warped_binary = luv_lab_filter(undist_img)
    ret = refine(warped_color, left_fit_last=None, right_fit_last=None)
    ret_new = refine_equidistant(undist_img, None, None, skip=False)
    left_fit, right_fit = ret["left_fit"], ret["right_fit"]
    left_fitx, right_fitx = ret["left_fitx"], ret["right_fitx"]
    left_lane_inds, right_lane_inds = ret["left_lane_inds"], ret["right_lane_inds"]
    nonzerox = ret["nonzerox"]

    left_fit_new, right_fit_new = ret_new["left_fit"], ret_new["right_fit"]
    left_fitx_new, right_fitx_new = ret_new["left_fitx"], ret_new["right_fitx"]

    print(len(left_lane_inds), len(right_lane_inds))
    print('left_fit,  left_fit_new')
    print(left_fit)
    print(left_fit_new)
    print('right_fit, right_fit_new')
    print(right_fit)
    print(right_fit_new)

    out_img = ret["out_img"]

    histogram = np.sum(warped_binary[hist_top_boundary:, :], axis=0)
    ploty = np.linspace(0, 720 - 1, 720)
    final = draw_area(undist_img, dst, src, left_fitx, right_fitx)

    plt.figure(figsize=(14, 8))
    plt.subplot(231)
    plt.imshow(img)
    plt.subplot(234)
    plt.imshow(warped_binary, cmap='gray')
    plt.title('l_ind={}, r_ind={}, non0x={}'.format(len(left_lane_inds), len(right_lane_inds), len(nonzerox)))
    plt.subplot(232)
    plt.plot(histogram)
    plt.xlim(0, 1280)

    plt.subplot(235)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow', linewidth=2)
    plt.plot(right_fitx, ploty, color='yellow', linewidth=2, label='Ori')
    plt.plot(left_fitx_new, ploty, color='fuchsia')
    plt.plot(right_fitx_new, ploty, color='fuchsia', label='Eq')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.legend(loc='upper left')
    plt.title('Left_A={:.5f}, Right_A={:.5f}'.format(left_fit[0], right_fit[0]))

    plt.subplot(233)
    plt.imshow(warped_color)
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='green')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.subplot(236)
    plt.imshow(final)
    plt.show()

    return


def eq_debug(ret, ret_new):
    left_fit, right_fit = ret["left_fit"], ret["right_fit"]
    left_fitx, right_fitx = ret["left_fitx"], ret["right_fitx"]
    left_lane_inds, right_lane_inds = ret["left_lane_inds"], ret["right_lane_inds"]

    left_fit_new, right_fit_new = ret_new["left_fit"], ret_new["right_fit"]
    left_fitx_new, right_fitx_new = ret_new["left_fitx"], ret_new["right_fitx"]
    l, r = left_fitx, right_fitx
    l_new, r_new = left_fitx_new, right_fitx_new
    ploty = np.linspace(0, 720 - 1, 720)
    plt.figure()
    plt.plot(l_new, -ploty, r_new, -ploty, color='red', linewidth=3, label='Equidistant')  # Equidistant
    plt.plot(l, -ploty, r, -ploty, color='green', linewidth=1, label='Original line')  # Original line
    plt.xlim(0,1280)
    plt.ylim(-720,0)
    plt.legend()

    return


def gamma_debug():
    img_dark = mpimg.imread('./test_images/test_ch6.jpg')
    img_light = mpimg.imread('./test_images/test_ch2.jpg')
    gam1 = exposure.adjust_gamma(img_dark, 0.3)
    gam2 = exposure.adjust_gamma(img_light, 2)

    plt.figure('adjust_gamma', figsize=(10, 4))
    plt.subplot(221)
    plt.title('origin dark image')
    plt.imshow(img_dark)
    plt.axis('off')

    plt.subplot(222)
    plt.title('gamma=0.3')
    plt.imshow(gam1)
    plt.axis('off')

    plt.subplot(223)
    plt.title('origin light image')
    plt.imshow(img_light)
    plt.axis('off')

    plt.subplot(224)
    plt.title('gamma=2')
    plt.imshow(gam2)
    plt.axis('off')
    plt.show()


def draw_area(undist, dst, src, left_fitx, right_fitx):
    img_shape = (720, 1280)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros(img_shape[0:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, 720 - 1, 720)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def draw(img):
    warped = warp(img)
    im1 = img2binary(warped)
    im2 = color_filter(warped, r_th=150, g_th=130, b_th=100)
    im3 = luv_lab_filter(warped, l_thresh=(210, 255)
)

    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.imshow(warped)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(im1, cmap='gray')
    plt.title('S Channel & Gradient')
    plt.subplot(223)
    plt.imshow(im2, cmap='gray')
    plt.title('RGB Filter')
    plt.subplot(224)
    plt.imshow(im3, cmap='gray')
    plt.title('LUV & LAB filter')

    plt.show()
    return


def apply_thresholds(image, show=True):
    img = warp(image)

    s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, 2]

    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # color_binary = np.dstack((u_binary, s_binary, l_binary))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    if show:
        # Plotting thresholded images
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(10, 4))
        f.tight_layout()

        ax1.set_title('Original Image', fontsize=16)
        ax1.imshow(image)
        ax2.set_title('Warped Image', fontsize=16)
        ax2.imshow(img)

        ax3.set_title('s binary threshold', fontsize=16)
        ax3.imshow(s_binary, cmap='gray')

        ax4.set_title('b binary threshold', fontsize=16)
        ax4.imshow(b_binary, cmap='gray')

        ax5.set_title('l binary threshold', fontsize=16)
        ax5.imshow(l_binary, cmap='gray')

        ax6.set_title('Combined color thresholds', fontsize=16)
        ax6.imshow(combined_binary, cmap='gray')
    else:
        return combined_binary

hist_top_boundary = 100  # boundary value of y axis
hist_left_start = 200
hist_right_start = 800
# Calculate approximated equidistant to a parabola
EQUID_POINTS = 25  # Number of points to use for the equidistant approximation
IMAGE_H = 720
pol = np.array([100, -50, 750])

ploty = np.linspace(0, 720 - 1, 720)


if __name__ == "__main__":
    img_size = [1280, 720]  # width, height

    dist_pickle = "./wide_dist_pickle.p"
    with open(dist_pickle, mode="rb") as f:
        CalData = pickle.load(f)
    mtx, dist = CalData["mtx"], CalData["dist"]

    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [0, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])

    img = mpimg.imread('./test_images/test_ch10.jpg')
    # debug_pipeline(img)

    # pol_eq = equidistant(pol, -90, plot=True)
    # print(pol_eq)


