Embedded Systems - Obstacle Detection

Create folders in directory: output, videos, mask-rcnn-coco

mask-rcnn-coco includes: 
  - colors.txt
  - frozen_inference_graph.pb
  - mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
  - object_detection_classes_coco.txt

videos folder includes:
  - Any .mp4 video that you would like to run analysis detection on
 


Now to run the code from command prompt:

1.	Navigate to the folder that contains the script(rcnn_video) and the following folders: (output, videos, mask-rcnn-coco)
2.	In command write:
Python rcnn_video.py –i videos\CarCrashGTA.mp4 –o output\GTA.mp4 -m mask-rcnn-coco 
