**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the **camera calibration** matrix and distortion coefficients given a set of chessboard images.
* Apply a **distortion correction** to raw images.
* Use color transforms to create a **thresholded binary** image.
* Apply a **perspective transform**("birds-eye view") to rectify binary image .
* Detect lane pixels and **polynominal fit** to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[https://github.com/uranus4ever/Advanced-Lane-Detection/blob/master]: # (Image References)

[image1]: ./Images/DistortionCorrection_CameraCalibration.png "Undistorted Chessboard"
[image2]: ./Images/DistortionCorrection_RawImage.png "Undistorted Raw Image"
[image3]: ./Images/ImageProcess.png "Binary"
[image4]: ./Images/Perspective_binary.png "Perspective Transform"
[image5]: ./Images/histogram.png "Histogram"
[image6]: ./Images/Perspective_binary.png "Perspective Binary"
[image7]: ./Images/SlidingWindow.png "SlidingWindow"
[image8]: ./Images/workflow.png "WorkFlow"
[video1]: ./output_videos/project.mp4 "Video"
[gif1]: ./Images/project.gif "Project_Video_gif"
[video2]: ./output_videos/challenge.mp4 "Video2"
[gif2]: ./Images/challenge.gif "challenge_gif"
[video3]: ./output_videos/harder_challenge.mp4 "Video3"
[gif3]: ./Images/harder_challenge.gif "harder_challenge_gif"

## Overview

This is the advanced project of [my lane detection project](https://github.com/uranus4ever/Lane-Detection). It aims to detect the driving lane in various environment in videos and calculate lane curvature. The output looks like the following gif:

| Normal Vidieo   | Challenge Video  | Harder Challenge Video |
|:---------:|:-----------:| :------: |
| ![alt text][gif1]   | ![challenge gif][gif2] | ![harder challenge gif][gif3] |
| ![video1][video1]   | ![video2][video2] | ![video3][video3]

## Usage
`project.py` contains source code to process the video.

### Dependencies

 - cv2
 - Numpy
 - collections
 - moviepy.editor

---

### Camera Calibration

The chessboard corners are the reference to generate `objpoints` and `imgpoints`.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. **Distortion Correction**

Matrix `mtx` and `dist` from camera calibration are applied to distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. **Color Transformation**

This part is the key to the lane detection. Inspired by Udacity cource and [Justin Heaton](https://github.com/JustinHeaton/Advanced-Lane-Finding), I tried RGB, HLS, LUV and LAB color space, in addition, gradient and magnitude, and their combination. As you can see from the following comparison, RGB filter is very sensitive to the threshold and could not split lanes from lightful environment. S channel and gradient combination could split lines on the road, but too much disturbance also left.

Finally I found color space transformation method as last figure shows. B channel from LAB space did a fairly good job to identify yellow lanes while L channel from LUV space could detect white lanes.

![alt text][image3]

#### 3. **Bird-eye Perspective Transformation**

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image.

![alt text][image6]

#### 4. **Lane Detection and Polynominal Fit**

Firstly, I calculated the histogram of non-zero x-axis in binary image. And based on the maximum sum of x position, I used sliding window search method to identify lane pixels. If it is previously detected in last frame, a quick search could be applied based on last detected x/y pixel positions with a proper margin.

![alt text][image5]

Then I fitted lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

#### 5. **Curvature and Position Calculation**

To connect pixel unit with real world meter unit, I defined conversions in x and y from pixels space to meters. In order to calculate precisely, I used detected lane width dynamically.
```
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / abs(xleft_eval - xright_eval)  # meters per pixel in x dimension
xmean = np.mean((xleft_eval, xright_eval))
offset = (img_shape[1]/2 - xmean) * xm_per_pix  # +: car in right; -: car in left side

fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / 
                np.absolute(2 * fit_cr[0])
```


#### 6. **Identified Lane Transformation back**

Last step is to transform the lane identified back onto the road image. Here is an example of my result of work flow:

![alt text][image8]

---

### Video Pipeline

There are various difficult environment conditions in the video, such as shadows, brightful / dirty road surface and confusion of new and old lane marks. To combat with all these conditions and make my algorithm robust, a `Line Class` is created to record and interact key line info among lastest multiple frames, which includs polynominal fit, x and y pixels and, more importantly, whether line is detected in the last frame.

---

### Reflection

#### Shortcomings

I spent lots of time on selecting proper color space and applicable threshold for finetuning. For now in `project.mp4` and `challenge.mp4` videos, the lane detection is stable and correctly. However, it is still far from robust to deal with harder_challenge video, in which the light differs extremely and part of lane is invisible for a few seconds. Recalling the behavioral cloning and transfer learning benefits with deep learning neural networks, I missed "smart brain" so much, which can be adaptive to various conditions and automatically self-fit.

#### Next Plan

1. Apply algorithm to my own camera recordings.
2. Combine deep learning thought with computer vision pipeline.