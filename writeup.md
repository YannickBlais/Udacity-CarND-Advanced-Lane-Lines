**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original Image"
[image2]: ./output_images/undistort_output.png "Undistorted"
[image3]: ./test_images/test1.jpg "Road Transformed"
[image4]: ./output_images/hls_binary_941.png "Binary Example"
[image5]: ./output_images/H_channel_1047.png "H Channel"
[image6]: ./output_images/L_channel_1047.png "L Channel"
[image7]: ./output_images/hls_binary_1047.png "Shadow Removed"
[image8]: ./output_images/original_unwarped.jpg "Original Unwarped"
[image9]: ./output_images/warped.jpg "Warped"
[image10]: ./output_images/histogram.png "Histogram"
[image11]: ./output_images/lanes_900.png "Sliding Windows"
[image12]: ./output_images/fit.png "Second Order Polynomial Fit"
[image13]: ./output_images/result_934.png "Final Result on a Random Image"

[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function perform_camera_calibration() located in "./find_lines.py", lines XXX through XXX.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `single_objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image
![alt text][image1]
Undistorted Image
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Color thresholding steps at function hls_select(), lines XXX through XXX in `find_lines.py` and gradient threshold steps are at function abs_sobel_thresh(), lines XXX through XXX in `find_lines.py`. I perform a final bitwise 'or' between the 2 binary results to merge them together. Here's an example of my output for this step.

![alt text][image4]

The color thresholding transform the image into HLS components and specific thresholds are used for each channel. I used the S channel that is good most of the time at finding both yellow and white lines, the L channel (which is similar to grayscale) with a simple theshold very good at finding the white lines. I also used the H channel to detect shadows on the road. This was rather useful in an area where trees were causing shadow on the road and confusing S channel line detection. Here's an example of H ans L channels for comparison:

H Channel
![alt text][image5]
L Channel
![alt text][image6]

Therefore I used the H Channel to subtract the shadow from the image and the final result looks like this:
Shadow Removed
![alt text][image7]
Note that the H channel was good at picking up shadows and sudden changes in lighting in this particular case, but may not be viable with all case scenarios, more testing should be done.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines XXX through XXX in the file `find_lines.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
      src = np.float32([[250, 688], [590, 459], [703, 459], [1053, 688]])
      dst = np.float32([[200, 720], [200, 0], [1000, 0], [1000, 720]])
```

Which can be beter visualized in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Undistorted Image with Source Points Drawn
![alt text][image8]
Warped Results with Destination Points Drawn
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lanes are first found from using the binarized image histogram projected on the x-axis, which gives us a good starting point to start looking for the lane lines from the bottom of the image. This step is performed in find_lines.py, finde_lines() function, lines XXX through XXX. The 2 highest peak of this histogram should be where the lanes are, here is an example of this histogram:

Binarized Images X-Axis Projection
![alt text][image10]

Following this first detection with the histogram, sliding windows along the Y-Axis are used to continue to detect the left and right lane lines. An example of the sliding windows is shown here:
![alt text][image11]

However, when lane lines were already found, we do not need to perform all the previous steps, but we can use this as a hint to "look" nearby and try to find the current lines. This is done in function find_lines_from_known_lines() in find_lines.py, lines XXX throught XXX. If finding lines from known lane lines does not work, a full lane lines detection with histogram and sliding windows (find_lines() again) is performed.

Following this step, function measure_curvature() in lines XXX through XXX in file `find_lines.py` fits a second order polynomial of the form x = A * y^2 + B * y + C on either lane lines. Here is an example of such a fit (including sliding windows):
![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the same function as the polynomial fit, which was measure_curvature() in lines XXX through XXX in file `find_lines.py`. The polynomial is first converted to world space using the pixel to meter ratio. Those ratios were approximated using the US standard 3.7 meters width per lane and the 720 pixels high were about 30 meters. The radius of curvature is then given by the following equation: R =(1+(2Ay+B)^2)^(3/2)) / ∣2A∣, where 'A' and 'B' are the polynomial fit coefficients.

Ultimately, the vehicle position to the center of the lane, the radius of curvature and the left and right fits are stored in a class "Line" (see find_lines.py, lines XXX throught XXX) that keeps track of those measurement and is reinitialized when finding lines from known lines fails, i.e. the current fit is too far from previous fit. All the mentioned metrics are averaged over at most maximum the last 10 measurements.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the main(), lines XXX through XXX in my code in `find_lines.py`. Here is an example of my result on an image from the "project_video.mp4" which displays radius of curvature, position from center of lane and frame id (Not that I display [Inf] when the radius is large in a straight line as the values are a bit meaningless at that point):

![alt text][image13]

---
### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

First, the camera calibration: I did not encounter major issues during this step as images were as expected, the only exception is the first image that did not display all the corners and was rejected for calibration. It was a good image to verify that calibration worked well however.

Second, binarization step: as discussed previously in this document, removing low-value pixels from the H-Channel may not be usable in all conditions. Maybe simply using adaptative histogram equalization (e.g. OpenCV's CLAHE) could help in some way to remove the effects from shadows.

Third, image unwarping: This step is also simple and straightforward and I did not encounter major issues.

Fourth, Polynomial fit: This step could benefit from a robust stochastic fit approach such as the Least Median of Square or RANSAC. This fit is particularly vulnerable to outliers.
