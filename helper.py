import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg


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
    Use cv2.undistort to 
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


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


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


def sliding_window(img):
    """
    Sliding window search for left and right lane.
    :param img: birds-eye perspective binary image
    :return: ret dictionay including fit 
    """
    # Assuming you have created a warped binary image called "binary_warped"
    binary_warped = img
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] // 2):, :], axis=0)
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
    margin = 100
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

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
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


def skip_sliding_window(img, left_fit, right_fit):
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

    ret = {'left_fit': left_fit,
           'right_fit': right_fit,
           'left_fitx': left_fitx,
           'right_fitx': right_fitx,
           'nonzerox': nonzerox,
           'nonzeroy': nonzeroy,
           'left_lane_inds': left_lane_inds,
           'right_lane_inds': right_lane_inds}

    return ret


def visualize():
    """Plot the images for README"""
    # Plot the Perspective Transformation
    image = mpimg.imread('./test_images/straight_lines2.jpg')
    plt.figure(figsize=(8, 2))
    plt.subplot(121)
    plt.imshow(image)
    draw_lines(src)
    plt.title('Original Image', fontsize=12)
    plt.subplot(122)
    plt.imshow(warp(image))
    draw_lines(dst)
    plt.title('Warped image with dest. points', fontsize=12)

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


def combine_bin(img):
    bin = img2binary(img)
    bin_color = color_filter(img)
    com_bin = np.zeros_like(bin_color)
    com_bin[(bin == 1) & (bin_color == 1)] = 1

    return com_bin


def debug_img():
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

    # (3,1) figure
    i = 1
    for image in glob.glob('./test_images/test_ch*.jpg'):
        plt.figure(figsize=(4, 7))
        img = mpimg.imread(image)
        img1 = img2binary(img)
        img2 = color_filter(img)
        img3 = combine_bin(img)

        plt.subplot(4,1,i)
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(4,1,i+1)
        plt.imshow(img1, cmap='gray')
        plt.axis('off')

        plt.subplot(4,1,i+2)
        plt.imshow(img2, cmap='gray')
        plt.axis('off')

        plt.subplot(4,1,i+3)
        plt.imshow(img3, cmap='gray')
        plt.axis('off')


if __name__ == "__main__":
    # image = mpimg.imread('./test_images/test_ch5.jpg')
    img_size = [1280, 720]  # width, height
    src = np.float32(
        [[575, 460],
         [283.33, 660],
         [1026.66, 660],
         [710, 460]])
    dst = np.float32(
        [[395, 0],
         [395, 660],
         [955, 660],
         [955, 0]])