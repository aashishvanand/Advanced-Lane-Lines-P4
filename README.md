# Advanced-Lane-Lines-P4
Self-Driving Car Nanodegree

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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First we took the chessboard (9x6) and found out the corners using cv2.findChessboardCorners() function the detailed output and the code could be found in the Advacned_Lane_Finding.pynb or Advanced_Lane_Finding.html<br>

Method : Read the image<br>
Convert into grayscale (cv2.COLOR_BGR2GRAY)<br>
Apply cv2.findChessboardCorners(gray, (9,6), None) <br>
Add points if found based on objp and corners <br>
Show the output <br>

Sample Outputs:

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/chessboard_1.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/chessboard_2.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/chessboard_3.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/chessboard_4.png"/><br>

###Pipeline (single images)

We now corrected distortion using cv2.undistort() using the chessboard first and then we follow the same for the test_images too the detailed output and the code could be found in the Advacned_Lane_Finding.pynb (In [5],In [6],In [7]) or Advanced_Lane_Finding.html<br>

Method: Read the image<br>
Run a cv2.calibrateCamera(objectpoints, imagepoints, img_size,None,None)<br>
Now use cv2.undistort(img, mtx, dist, None, mtx)<br>
Show the Output<br>

Sample Outputs:<br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/undistorted_chess.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/test_image_1.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/test_image_2.png"/><br>

###Perspective Transform (birds-eye view)

Actually its asked to do a perspective transformation of the combined binary output. But i  felt doing a perspective transformation (birds-eye view) and finding the binary would even reduce the work. In finding thresholds and finding combined binary output only the road and the line markings will be the input so there will be no further noise in the image. <br>

Method : Read the image<br>
Run undistort_view() function to get a distorction free image<br>
src = np.float32([[490, 482],[810, 482],[1250, 720],[40, 720]])<br>
dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])<br>
Use to generate cv2.getPerspectiveTransform(src, dst)<br>
Show the output<br>

Sample Output:<br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/helicopter_view_1.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/helicopter_view_2.png"/><br>

###Combined Thresholded Binary Image
apply_thresholds() is the function that uses three diffrent channels s_channel(Saturation), l_channel(Luminance), and b_channel(for yellow color) Most of the lane will fall under this catagory and a binary threshold can be generated leaving out other factors. It was able to find both Yellow and White lane lines and often gets distracted by shadow in the road. The limts are
The S Channel from the HLS color space, using cv2.COLOR_BGR2HLS function
min threshold 180 <br>
max threshold 255 <br>
<br>

The L Channel from the LUV color space, using cv2.COLOR_BGR2LUV function. It was able to find the white lines fully but it ignored the yellow lines fully. The limits are.
min threshold 215 <br>
max threshold 255 <br>
<br>

The B channel from the Lab color space, using cv2.COLOR_BGR2Lab function. It was able to find the yellow lines fully but it ignored the white lines fully.<br>
min threshold 145 <br>
max threshold 200 <br>
<br>
The L channel and the B channel together made sure that all the lane lines (yellow and white) are included. S channel was left because it added more noise to the combaind binary thresholded image.

Method : When the image is passed<br>
it is split into diffrent channels namely s channel, l channel and the b channel based on the threshold.<br>
combained output is

Sample Output: <br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/combined_1.png"/><br>

####Detect lane pixels and fit to find the lane boundary.

The fitting of polynomial to each line was done by
<ul>
<li>Identifying peaks in a histogram of the image to determine location of lane lines.</li><br>
<li>Identifying all non zero pixels around histogram peaks using the numpy function numpy.nonzero().</li><br>
<li>Fitting a polynomial to each lane using the numpy function numpy.polyfit().</li><br>
</ul>
####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
