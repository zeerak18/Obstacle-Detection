Embedded Systems - Obstacle Detection

Create folders in directory: output, videos, mask-rcnn-coco

mask-rcnn-coco includes: 
  - colors.txt
  - frozen_inference_graph.pb
  - mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
  - object_detection_classes_coco.txt

videos folder includes:
  - Any .mp4 video that you would like to run analysis detection on
  - Sample videos are included
 


Now to run the code from command prompt:

1.	Navigate to the folder that contains the script(rcnn_video) and the following folders: (output, videos, mask-rcnn-coco)
2.	In command write:
Python rcnn_video.py –i videos\CarCrashGTA.mp4 –o output\GTA.mp4 -m mask-rcnn-coco 


General Description:

•	Import the necessary packages (numpy, argparse, imutils, time, OpenCV, os  matplotlib and math)
•	construct  the argument parse and parse the arguments
•	load the COCO class labels our Mask R-CNN was trained on
•	initialize a list of colors to represent each possible class label
•	derive the paths to the Mask R-CNN weights and model configuration
•	load our Mask R-CNN trained on the COCO dataset (90 classes) from disk
•	Initialize the video stream and pointer to output video file
•	Determine the total number of frames in the video file
•	Frames Skip factor set here to 3, if you inrease it will neglict more frames
•	loop over frames from the video file stream
•	Read the next frame from the file
•	Initialize a big mask that will contains all objects masks For Collision detection
•	Initialize list to get object's center for all detected objects within a frame
•	if the frame was not grabbed, then we have reached the end of the stream
•	construct a blob from the input frame and then perform a forward pass of the Mask R-CNN, giving us (1) the bounding box
•	Coordinates of the objects in the image along with (2) the pixel-wise segmentation for each specific object
•	loop over the number of detected objects
•	extract the class ID of the detection along with the confidence (i.e., probability) associated with the prediction
•	filter out weak predictions by ensuring the detected probability is greater than the minimum probability
•	scale the bounding box coordinates back relative to the size of the frame and then compute the width and the height of the bounding box
