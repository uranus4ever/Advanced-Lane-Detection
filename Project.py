import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from helper import warp, draw_lines, undistort, sliding_window, combine_bin


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
        self.previousfit = [np.array([False])]
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.
    def get_fit(self):
        return self.best_fit
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


def draw_area(undist, warped_size, dst, src, left_fit, right_fit):
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
    curv = [np.mean((left_curverad, right_curverad)), left_curverad, right_curverad]
    # print(left_curverad, 'm', right_curverad, 'm', curv, 'm')

    xleft_eval = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    xright_eval = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    xmean = np.mean((xleft_eval,xright_eval))
    offset = (warped_size[1]/2 - xmean) * xm_per_pix # +: car in right; -: car in left side
    return curv, offset


def draw_curv(image):

    img_size = [1280, 720]
    img_shape = (720, 1280, 3)

    undist_img = undistort(image, mtx, dist)
    binary = combine_bin(undist_img)
    warped = warp(binary)
    ret = sliding_window(warped)
    left_fit, right_fit = ret['left_fit'], ret['right_fit']
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # warp_zero = np.zeros(warped.shape).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # pts = np.hstack((pts_left, pts_right))
    # cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    warped_color = warp(undist_img)
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='green')

    return warped_color


def process_image(image):

    global mtx, dist
    global left_line, right_line

    img_size = [1280, 720]
    img_shape = (720, 1280, 3)

    undist_img = undistort(image, mtx, dist)
    binary = combine_bin(undist_img)
    warped = warp(binary)

    if not (left_line.detected or right_line.detected):
        ret = sliding_window(warped)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)

        line_curv, offset = curvature(img_shape, left_fit, right_fit)

        left_line.detected = True
        right_line.detected = True
    else:
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()

        line_curv, offset = curvature(img_size, left_fit, right_fit)
        # left_curvature - right_curvature diff too much or left/right curvature is too small
        if (abs(line_curv[1] - line_curv[2]) > 1000) or (line_curv[1] < 600) or (line_curv[2] < 600):
            ret = sliding_window(warped)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            left_line.previousfit = [np.array([])]
            right_line.previousfit = [np.array([])]
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            line_curv, offset = curvature(img_shape, left_fit, right_fit)
        else:
            retSkip = SkipSlidingWindow(warped, left_fit, right_fit)
            # Update only when line detected in current frame
            if retSkip is not None:
                left_fit = retSkip['left_fit']
                right_fit = retSkip['right_fit']
                left_fit = left_line.add_fit(left_fit)
                right_fit = right_line.add_fit(right_fit)
                # +: car in right; -: car in left side
                line_curv, offset = curvature(img_shape, left_fit, right_fit)
            else:
                left_line.detected = False
                right_line.detected = False

        ret = SkipSlidingWindow(warped, left_fit, right_fit)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']

        # Update only when line detected in current frame
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            # +: car in right; -: car in left side
            line_curv, offset = curvature(img_shape, left_fit, right_fit)
        else:
            left_line.detected = False
            right_line.detected = False


    result = draw_area(undist_img, img_shape, dst, src, left_fit, right_fit)

    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    text1 = 'Radius of Curvature = %d(m), l=%d(m), r=%d(m)'
    text2 = 'Vehicle is %.2f(m) left of center'

    cv2.putText(result, text1 % (int(line_curv[0]), int(line_curv[1]), int(line_curv[2])),
                              (60, 100), font, 1.0, (255, 255, 255), thickness=2)
    cv2.putText(result, text2 % (-offset),
                              (60, 130), font, 1.0, (255, 255, 255), thickness=2)
    return result



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

# import Camera Calibration Parameters
dist_pickle = "./wide_dist_pickle.p"
with open(dist_pickle, mode="rb") as f:
    CalData = pickle.load(f)
mtx, dist = CalData["mtx"], CalData["dist"]
frame = 10  # latest frames number of good detection
# left_line = Line(n=frame)
# right_line = Line(n=frame)



# TODO: 3rd polynominal. equidistant
'''
#Calculate approximated equidistant to a parabola
EQUID_POINTS = 25 # Number of points to use for the equidistant approximation
def equidistant(pol, d, max_l = 1, plot = False):
    y_pol = np.linspace(0, max_l, num=EQUID_POINTS)
    x_pol = pol_calc(pol, y_pol)
    y_pol *= IMAGE_H # Convert y coordinates to [0..223] scale
    x_m = []
    y_m = []
    k_m = []
    for i in range(len(x_pol)-1):
        x_m.append((x_pol[i+1]-x_pol[i])/2.0+x_pol[i]) # Calculate polints position between given points
        y_m.append((y_pol[i+1]-y_pol[i])/2.0+y_pol[i])
        if x_pol[i+1] == x_pol[i]:
            k_m.append(1e8) # A vary big number
        else:
            k_m.append(-(y_pol[i+1]-y_pol[i])/(x_pol[i+1]-x_pol[i])) # Slope of perpendicular lines
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    k_m = np.array(k_m)
    #Calculate equidistant points
    y_eq = d*np.sqrt(1.0/(1+k_m**2))
    x_eq = np.zeros_like(y_eq)
    if d >= 0:
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
    pol_eq = np.polyfit(y_eq, x_eq, len(pol)-1) # Fit equidistant with a polinomial
    if plot: #Visualize results
        plt.plot(x_pol, y_pol, color='red', linewidth=1, label = 'Original line') #Original line
        plt.plot(x_eq, y_eq, color='green', linewidth=1, label = 'Equidistant') #Equidistant
        plt.plot(pol_calc(pol_eq, y_pol), y_pol, color='blue',
                 linewidth=1, label = 'Approximation') #Approximation
        plt.legend()
        for i in range(len(x_m)):
            plt.plot([x_m[i],x_eq[i]], [y_m[i],y_eq[i]], color='black', linewidth=1) #Draw connection lines
        plt.savefig('readme_img/equid.jpg')
    return pol_eq

pol = np.array([106.65796008,  -49.57665396,  718.87055435])
print(equidistant(pol, -90, plot=True))
'''


# video_output = './output_videos/challenge_output.mp4'
# gif_output = './output_videos/challenge_output.gif'
#
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
#
# input_path = './test_videos/challenge_video.mp4'
#
# clip1 = VideoFileClip(input_path).subclip(10,16)
# # clip1 = VideoFileClip(input_path)
# final_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#
# # final_clip.write_gif(gif_output, fps=25)
# final_clip.write_videofile(video_output, audio=False)


