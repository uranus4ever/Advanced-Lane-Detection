**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the **camera calibration** matrix and distortion coefficients given a set of chessboard images.
* Apply a **distortion correction** to raw images.
* Use color transforms, gradients, etc., to create a **thresholded binary** image.
* Apply a **perspective transform**("birds-eye view") to rectify binary image .
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Images/DistortionCorrection_CameraCalibration.png "Undistorted Chessboard"
[image2]: ./Images/DistortionCorrection_RawImage.png "Undistorted Raw Image"
[image3]: ./Images/CombineColorandGradient.png "Binary-Combine Color and Gradient"
[image4]: ./Images/PerspectiveTransform.png "Perspective Transform"
[image5]: ./Images/histogram.png "Histogram"
[image6]: ./Images/Perspective_binary.png "Perspective Binary"
[image7]: ./Images/SlidingWindow.png "SlidingWindow"
[image8]: ./Images/workflow.png "WorkFlow"
[video1]: ./project_video.mp4 "Video"
[gif1]: ./Images/project.gif "Project_Video_gif"
[video2]: ./ "Video2"
[gif2]: ./Images/ "challenge_gif"
[video3]: ./ "Video3"
[gif3]: ./Images/ "harder_challenge_gif"

## Overview

Detect the driving lane on complex condition in dynamic videos and calculate lane curvature. The output looks like the following gif:

| Project   | Challenge   | 
|:---------:|:-----------:| 
| ![alt text][gif1]   | ![challenge gif][gif2] | 
| ![video1][video1]   | ![video2][video2] |

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficienst. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in  lines of the file called `helper.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I aApplythea distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.
  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 395, 0        | 
| 283, 660      | 395, 660      |
| 1026, 660     | 955, 660      |
| 710, 460      | 955, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

![alt text][image6]

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

Define conversions in x and y from pixels space to meters
```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
![alt text][image5]

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # trhouge # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:



---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

![alt text][image8]

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
