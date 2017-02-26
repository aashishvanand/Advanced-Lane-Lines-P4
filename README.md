# Advanced-Lane-Lines-P4
Self-Driving Car Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# Camera Calibration

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

# Pipeline (single images)

We now corrected distortion using cv2.undistort() using the chessboard first and then we follow the same for the test_images too the detailed output and the code could be found in the Advacned_Lane_Finding.pynb (In [5],In [6],In [7]) or Advanced_Lane_Finding.html<br>

Method: Read the image<br>
Run a cv2.calibrateCamera(objectpoints, imagepoints, img_size,None,None)<br>
Now use cv2.undistort(img, mtx, dist, None, mtx)<br>
Show the Output<br>

Sample Outputs:<br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/undistorted_chess.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/test_image_1.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/test_image_2.png"/><br>

# Perspective Transform (birds-eye view)

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

# Combined Thresholded Binary Image
apply_thresholds() is the function that uses three diffrent channels s_channel(Saturation), l_channel(Luminance), and b_channel(for yellow color) Most of the lane will fall under this catagory and a binary threshold can be generated leaving out other factors. It was able to find both Yellow and White lane lines and often gets distracted by shadow in the road. The limts are
The S Channel from the HLS color space, using cv2.COLOR_BGR2HLS function<br>
min threshold 180 <br>
max threshold 255 <br>
<br>

The L Channel from the LUV color space, using cv2.COLOR_BGR2LUV function. It was able to find the white lines fully but it ignored the yellow lines fully. The limits are.<br>
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
combained binary is taken as a output

Sample Output: <br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/combined_1.png"/><br>

# Detect lane pixels and fit to find the lane boundary.

The fitting of polynomial to each line was done by
<ul>
<li>Identifying peaks in a histogram of the image to determine location of lane lines.</li><br>
<li>Identifying all non zero pixels around histogram peaks using the numpy function numpy.nonzero().</li><br>
<li>Fitting a polynomial to each lane using the numpy function numpy.polyfit().</li><br>
</ul>
####Radius of Curvature of the lane and the position of the vehicle with respect to center.

Method:
Calculated the average of the x intercepts from each of the two polynomials position = (rightx_int+leftx_int)/2<br>
Calculated the distance from center by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis distance_from_center = abs(image_width/2 - position)<br>
If the horizontal position of the car was greater than image_width/2 than the car was considered to be left of center, otherwise right of center.<br>
Finally, the distance from center was converted from pixels to meters by multiplying the number of pixels by 3.7/700.<br>

# Measure Radius of Curvature for each lane line
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])

Radius of Curvature was taken as an average of left and right curves. 

# Calculate the position of the vehicle
    center = abs(640 - ((rightx_int+leftx_int)/2))
    
    offset = 0 
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])
    pts = np.hstack((pts_left, pts_right))

# Warp the detected lane boundaries back onto the original image

The polynomial plotted image is shown in the left and the image in the right shown the fill the space between the polynomials to highlight the lane that the car is in another perspective transformation is used to unwrap the image from helicopter view or top down view to its original presoective. The vehicle distance from the center and the radius of curvature is printed.<br>

Output<br>
<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/fill_lanes_1.png"/><br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/fill_lanes_2.png"/><br>


# Pipeline (video)

The final step was to expand the pipeline which was earlier done for still images to process videos frame-by-frame, This could stimulate a realtime situation. Video is still based on combining images like 25fps or 30fps. <br>

To create a smooth output a class was created for left and right lane lines and stored features of each lane for averaging across frames.

Based on the previous frame the algo works. first it check whether lane was detected in previous frame. If it was, then it only checks for lane pixels in near by to the polynomial calculated in the previous frame. This way scanning of entire image is reduced and the pixels detected can have a high confidence of belonging to the lane line because they are based on the location of the lane in the previous frame and not somewhere else usually these lanes are straight unless its<br>

<img src="https://raw.githubusercontent.com/aashishvanand/Advanced-Lane-Lines-P4/master/output_images/fail.jpg"/><br>
ha ha thats unpractical. :P <br>

If no lanes are detected in the previous frame. it will scan the entire binary image for nonzero pixels to represent the lanes.

In order to make the output smooth I chose to average the coefficients of the polynomials for each lane line over a span of 10 frames. 

Output:<br>
Project Video:<br>
[![Final Output](http://img.youtube.com/vi/1VZV8GxgNNk/0.jpg)](https://www.youtube.com/watch?v=1VZV8GxgNNk)<br>
<br>
Challenge Result:<br>
[![Final Output](http://img.youtube.com/vi/VSJUjQqW8Y4/0.jpg)](https://www.youtube.com/watch?v=VSJUjQqW8Y4)<br>
<br>
And the epic fail Harder Challenge Result:<br>
[![Final Output](http://img.youtube.com/vi/iMOMkybZ1vo/0.jpg)](https://www.youtube.com/watch?v=iMOMkybZ1vo)<br>
<br>

# Discussion

When there is heavy shadow like the one in the Harder Challenge Result algo fails. What i felt was further experimenting channels or a combained output of the normal camera with a ir camera can help to solve this issue. Weather Conditions play a major role however better our algorithm is.<br>
Just with the diffrence of 10 in the b_channel can make the prediction worse. so only optimised paratmets should be used. Experimenting with luminance converting into greyscale and mapping the lanes as a second refrence as a backup could further give some better output in the final video. Due to the deadline and time constraints i am submitting it.
