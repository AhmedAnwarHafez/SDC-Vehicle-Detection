##Writeup Template


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-not-car.png
[image2]: ./output_images/HOG-comparison.png
[image3]: ./output_images/search-grid.png
[image4]: ./output_images/pipeline-test-images.png
[image5]: ./output_images/heat-blob.png
[image6]: ./output_images/heatmap.png
[image7]: ./output_images/final-result.png
[video1]: ./project_video.mp4



---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

After reading all the file names (vehicles and non-vehicles), `extract_features` is called. This method extracts 3 types of feaures, spatial, color and HOG features and it appears in the  5th cell in `Submission.ipynb`. The actual method that extracts the HOG feature is in the `functions.py` file line #15.


Here is a plot showing the HOG features of a vehicle and a non-vechicle using the Gray color space with `orient=9`, `pix_per_cell=8`, `cells_per_block=2` as parameters to the `skimage.hog()` method.

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I ran a coarse random search to find the best HOG parameters and tested them by training a Linear SVM and evaluated on a fixed testing set.
I found the following setting gives me the best results:

	color_space = 'YCrCb' 
	orient = 9  
	pix_per_cell = 8 
	cell_per_block = 2 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After normalizing, shuffling and splitting the data, I trained a Linear SVM (cell #7) and NN using keras (cell #8) and scored both models to see which one has the highest accuracy. The NN test accuracy was usually 1% higher than the SVM. My NN had a hidden layer with 4096 units and used ReLus as activation functions. I trained the model for 3 epochs.

The difference between LinearSVM and the NN classifier is not noticable in the final video.

Note: all plotted examples in this writeup were generated using a LinearSVM.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The method `find_cars()` in the same notebook in cell #9 searches for cars in the lower part of the image.
My approach uses a search grid of `scale=1.5` and 1 cell per step.

Here is an example of how it looks like:

![alt text][image3]

The trained classifer scores each cell in the grid by combining 3 features, color, spatial and HOG features. If the classifier predicts that there is a car, the position of the current cell is a added to a list and returned to the main pipeline.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried searching on 2 scales and concatenated the detected bboxes into a bigger numpy array.

    scale1 = 2.5
    scale2 = 1.5
    
    box_list1 = find_cars(img, ystart, ystop, scale1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    box_list2 = find_cars(img, ystart, ystop, scale2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
However, I settled calling the find_cars() once per frame with one scale of `1.5`.

Here is the end result of executing the piple line on 4 test images.

![alt text][image4]

###A few things to consider to increase the reliability of the Classifer and the Sliding Window:

1. The quality of the features that the classifer is trained on. For example, choosing the color histogram only would result in a sigfinicant decrease in the classification accuracy because classifying based on color is not enough. On the other hand, HOG features describe the shape of the samples and thus helps the classifier to distinguish between different objects. So appending more feature types such as color, HOG and spatial features to a classifier increase the accuracy.

2. Normalizing the data before training a classifier is also crucial, because some types of features have high magnitudes and other have smaller. This issue could lead that the classifier is sensitive to features with wide ranges than others and therefore affects the accuracy of the classifier. For example, color features range between 0 and 255 but features like HOG range between 0-1.

3. Testing the classifer on an unseen dataset to measure the actual accuracy of the classifier.

4. The sliding window scale defines the size of the window to search in a given frame. This parameter affects on detecting closer or distant objects. For example, a scale of `1` means that the search grid contains tiny cells as I already described in the question above.
5. Applying a rejection threshold on heatmaps to remove False Positives.
Often times, the Sliding Window technique detects False Positives and to combat this problem is to build a heat map and use rejection threshold. The rejection threshold is a parameter that can be tuned to get the best results. I used a value of `2` and it works well. 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_submission.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After getting the `box_list` from `find_cars()` method, the pipelinproject_submission

The returned value from find_cars() method in cell #10 is a list of bounding boxes detected by the classifier. This list variable contains False Positives and Duplicate detections. To solve this problem, I used a heat map in each frame. Places where bounding boxes intersect show hot spots or blobs. The more the detected boxes interesect, the more the center color changes to white. Here is an example

![alt text][image5]

The left blob indicates that there are many bounding boxes than right one.


This a code snippet from cell #10 in the notebook.

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)


After getting the heat map, I applied a threshold of magnititude of `2` is ued to remove any False Duplicates from the current Frame.

### Here are 2 frames and their corresponding heatmaps:

![alt text][image6]


I then used the `Label()` from Scipy package to identify the cars in the Frame. The method returns the number of objects and their location on the frame.
The last step in the pipeline is to draw a nice bounding boxes around each detected object. This step happens in cell #10 in `draw_labeled_bboxes()` which finds each pixel for each label and gets the corver locations to draw the final box. 


### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are two issues that I can see with this pipeline:

1. The poor performance is because the classifier predicts every cell in every frame. 
2. Distant cars are not detected. This could be  `find_cars()`resizes the images and losses important information that the classifier can't detect cars properly.

If I had the opporunity to implement and object detection, I would use YOLO or SSD. Both are robust, state-of-the-art techniques and both work in real-time. 