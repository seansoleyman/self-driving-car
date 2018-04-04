# Finding Lane Lines on the Road

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines function.

My pipeline consists of 7 steps:
1. Convert to grayscale.
2. Canny edge detection.
3. Region masking.
4. Hough transform.
5. Discard any line segments with slope that is outside a reasonable range.
6. Extrapolate remaining line segments so that they run from the top to the bottom of the region of interest.
7. Average the horizontal position at the top and at the bottom.

To improve robustness, the field of view is split up into two regions of interest - the left and right field of view. The draw_lines function is replaced with a function called find_lane_line, which works on the left and right sides of the image separately. When finding the left lane line, only lines with slope between 0.4 and 2.0 are included in the average. When finding the right lane line, only lines with slope between -2.0 and -0.4 are included in the average. After the two averaged lines are found, they are both drawn over the original image.

### 2. Identify potential shortcomings with your current pipeline

The most obvious shortcoming is that the algorithm may identify any long edge as a lane marking. This is evident in the middle of the "challenge" video, which includes shadows and an abrupt transition in the color of the road. At this point, the Hough transform returns an incoherent jumble of segments at various angles. In my solution, this causes the line to disappear momentarily since none of the jumbled segments has a plausible slope.

As far as the overall approach is concerned, it would be difficult to use this algorithm to distinguish between different types of lane lines. Also, the very nature of the Hough transform prevents it from being used on curved road markings.

### 3. Suggest possible improvements to your pipeline

Instead of using grayscale Canny edge detection, maybe it would be possible to use a color edge detection kernel that only senses white-gray and yellow-gray edges. Gaussian blurring could also be used to avoid detecting certain objects as lane lines, but could also make it difficult to detect lane lines that are far off in the distance.

It may be possible to re-cast this as a machine learning object detection problem, which could be solved using a convolutional neural net approach. Given enough training data, this approach could yield much more robust differentiation between lane lines and other objects.