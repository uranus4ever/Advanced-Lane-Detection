import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from helper import warp, undistort, sliding_window, skip_sliding_window, combine_bin


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, n):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # n is window size of the moving average
        self.n = n
        # Polynomial coefficients: x = A*y**2 + B*y + C
        self.A = []
        self.B = []
        self.C = []
        self.previousfit = [np.array([False])]
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.

    def get_fit(self):
        return self.best_fit

    def get_x(self, fit):
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        self.allx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        return self.allx

    def get_curv(self, fit):
        self.radius_of_curvature = curvature(fit)
        return self.radius_of_curvature

    def add_fit(self, fit):
        q_full = len(self.A)
        self.A.append(fit[0])
        self.B.append(fit[1])
        self.C.append(fit[2])
        self.previousfit.append(np.array([fit[0], fit[1], fit[2]]))
        self.current_fit = np.array([fit[0], fit[1], fit[2]])

        if q_full >= self.n:
            self.A.pop(0)
            self.B.pop(0)
            self.C.pop(0)
            self.previousfit.pop(0)

        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)
        self.best_fit = [self.A_avg, self.B_avg, self.C_avg]
        return self.best_fit


def draw_area(undist, dst, src, left_fitx, right_fitx):
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros(img_shape[0:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))

    # Combine the result with the original image
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


def curvature(fit):
    """
    calculate curvature from fit parameter
    :param fit: [A, B, C]
    :return: radius of curvature (in meters unit)
    """

    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * fit_cr[0])

    # curv = [np.mean((left_curverad, right_curverad)), left_curverad, right_curverad]
    # print(left_curverad, 'm', right_curverad, 'm', curv, 'm')

    return curverad


def car_pos(left_fit, right_fit):
    """
    Calculate the position of car on left and right lane base (convert to real unit meter)
    :param left_fit: 
    :param right_fit: 
    :return: distance (meters) of car offset from the middle of left and right lane
    """
    xleft_eval = left_fit[0] * np.max(ploty) ** 2 + left_fit[1] * np.max(ploty) + left_fit[2]
    xright_eval = right_fit[0] * np.max(ploty) ** 2 + right_fit[1] * np.max(ploty) + right_fit[2]
    xmean = np.mean((xleft_eval, xright_eval))
    offset = (img_shape[1]/2 - xmean) * xm_per_pix  # +: car in right; -: car in left side

    return offset


# def draw_curv(img):
#
#     undist_img = undistort(img, mtx, dist)
#     binary = combine_bin(undist_img)
#     warped = warp(binary)
#     ret = sliding_window(warped)
#     left_fit, right_fit = ret['left_fit'], ret['right_fit']
#     ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
#     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
#     # warp_zero = np.zeros(warped.shape).astype(np.uint8)
#     # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#     # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#     # pts = np.hstack((pts_left, pts_right))
#     # cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
#     warped_color = warp(undist_img)
#     plt.plot(left_fitx, ploty, color='green')
#     plt.plot(right_fitx, ploty, color='green')
#
#     return warped_color


def process_image(img):

    global mtx, dist, src, dst
    global left_line, right_line

    undist_img = undistort(img, mtx, dist)
    binary = combine_bin(undist_img)
    warped = warp(binary, src, dst)

    if (left_line.detected or right_line.detected) is False:
        ret = sliding_window(warped)
        left_fit, left_fitx = ret['left_fit'], ret["left_fitx"]
        right_fit, right_fitx = ret['right_fit'], ret["right_fitx"]
        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)

        left_line.detected = True
        right_line.detected = True

    else:  # both left and right line detected.
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()

        ret = skip_sliding_window(warped, left_fit, right_fit)
        left_fit, left_fitx = ret['left_fit'], ret["left_fitx"]
        right_fit, right_fitx = ret['right_fit'], ret["right_fitx"]
        left_curv = left_line.get_curv(left_fit)
        right_curv = right_line.get_curv(right_fit)

        # check the fit quality by skip_sliding_search
        if (abs(left_curv - right_curv) < 500) | ((left_curv > 2500) & (right_curv > 2500)):
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)

        else:  # decide to re-search
            left_line.detected = False
            right_line.detected = False
            left_fit = left_line.get_fit()
            right_fit = right_line.get_fit()

    left_curv = left_line.get_curv(left_fit)
    right_curv = right_line.get_curv(right_fit)
    mean_curv = np.mean([left_curv, right_curv])
    offset = car_pos(left_fit, right_fit)

    left_fitx = left_line.get_x(left_fit)
    right_fitx = right_line.get_x(right_fit)
    result = draw_area(undist_img, dst, src, left_fitx, right_fitx)

    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    text1 = 'Radius of Curvature = %d(m), l=%d(m), r=%d(m)'
    text2 = 'Vehicle is %.2f(m) left of center'

    cv2.putText(result, text1 % (int(mean_curv), int(left_curv), int(right_curv)),
                              (60, 100), font, 1.0, (255, 255, 255), thickness=2)
    cv2.putText(result, text2 % (-offset),
                              (60, 130), font, 1.0, (255, 255, 255), thickness=2)
    return result

img_shape = (720, 1280)
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
ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

# import Camera Calibration Parameters
dist_pickle = "./wide_dist_pickle.p"
with open(dist_pickle, mode="rb") as f:
    CalData = pickle.load(f)
mtx, dist = CalData["mtx"], CalData["dist"]
frame = 4  # latest frames number of good detection
left_line = Line(n=frame)
right_line = Line(n=frame)

# video_output = './output_videos/project_test.mp4'
# input_path = './test_videos/project_video.mp4'
#
# # clip1 = VideoFileClip(input_path)
# clip1 = VideoFileClip(input_path).subclip(30,45)
#
# final_clip = clip1.fl_image(process_image)
# final_clip.write_videofile(video_output, audio=False)

image = mpimg.imread('./test_images/test_ch5.jpg')
new = process_image(image)
plt.figure()
plt.imshow(new)

