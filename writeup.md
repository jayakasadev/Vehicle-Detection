**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images

I started by reading in all the `vehicle` and `non-vehicle` images from the `dataset` directory.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![non-vehicle][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/extra1.png]
![non-vehicle][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/image0000.png]

The `feature_extration` method takes in a file path list, loads the images and processes each one. I use this method to load all images from the `dataset` directory and prepare them for training an SVM later on.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

#### 2. HOG parameters.

I tried various combinations of parameters. The process took a lot of experimentation and many hours to see what worked best. Certain combinations were better than others. This method was really good at picking up lane lines, which I explored a little. However, in the end, I settled on using the parameters used during the lectures. 

![ROAD HOG][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/hog_road.JPG]

#### 3. Training a classifier.

I trained a `Linear SVM` using using a combination of HOG features, color space features and spatial binning. THese features were all scaled prior to training to prevent any one part from having too much influence. This was done in the `Training the SVM for image recognition` section.

### Sliding Window Search

#### 1. Implementing a Sliding Window Search.

I searched sequentially through the lower half of image. This was done because the top half contains only the sky. I searched through a range of scales from `[1 to 3]` in increments of `0.5`. This was done because cars get smaller as they get further, so it is important to search at all scales. This was all done iteratively, which was really slow and unweildy. 

After all vehicles were identified, at all scales, I corrected for multiple identification and false positives. This was done by adding `heat` and thresholding the image.

![Identified][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/identified.JPG]

#### 2. Optimization of Classifier.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Heatmap][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/heatmap.JPG]
![Heatmap][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/heatmap1.JPG]
![Heatmap][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/heatmap2.JPG]
---

### Video Implementation

#### 1. Final Video Output.
Here's a [link to my video result](https://github.com/jayakasadev/Vehicle-Detection/blob/master/project_video_complete.mp4)


#### 2. Filtering for false positives and Multiple Detections.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![Identified][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/identified.JPG]
![Heatmap][https://github.com/jayakasadev/Vehicle-Detection/blob/master/samples/heatmap.JPG]

---

### Discussion

#### Pitfalls, Shortcomings and Future Improvements
My pipeline was exceptionally slow. It took an hour for the pipeline to process the final video. That is extremely awful when working with real time video. Also, it would be way better to use a Deep Neural Network to identify vehicles, rather than using an SVM after creating feature vectors from images. Though the SVM trains really fast, the ETL process for it is rather tedious when compared to a using Convolutions in a Deep Neural Network.
