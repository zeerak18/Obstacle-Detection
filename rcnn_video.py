# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
import argparse
#import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import math
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video ")
ap.add_argument("-o", "--output", required=True,
	help="path to output video ")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability ")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold ")
args = vars(ap.parse_args())

# If you want to set values in the code without parsing Comment lines (17-28) and Uncomment (31-37)
#ap = argparse.ArgumentParser()
#args = vars(ap.parse_args())
#args["input"]="./videos/CarCrashGTA.mp4"
#args["output"]="./output/GTA1.mp4"
#args["mask_rcnn"]="./mask-rcnn-coco"
#args["confidence"]=0.5
#args["threshold"]=0.3


# load the COCO class labels our Mask R-CNN 
labelsDir = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsDir).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
ClassColors = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on COCO dataset
print("loading Mask R-CNN")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["input"])
writer = None

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("{} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("Could not determine # of frames in video")
	total = -1

# Frames Skip factor set here to 3, if you inrease it will neglict more frames
j = 0 
skipFactor =3

NoticeFrame=0
collisionFrame=0
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    vs.set(cv2.CAP_PROP_POS_FRAMES, j*skipFactor)
    (grabbed, frame) = vs.read()
    
    # Initialize a big mask that will contains all objects masks For Collision detection
    # Initialize list to get object's center for all detected objects whithin a frame
    CombinedMask = np.zeros((frame.shape[0],frame.shape[1]))
    ObjectsCenters = []
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
    	break

    # construct a blob from the input frame and then perform a
    # forward pass of the Mask R-CNN, giving us (1) the bounding box
    # coordinates of the objects in the image along with (2) the
    # pixel-wise segmentation for each specific object
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final",
    	"detection_masks"])
    end = time.time()
    AllMasks = []
    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID and probability
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        # filter out weak predictions 
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back and get width and height
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract object's mask, resize it to same dimensions of box with a binary threshold
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > args["threshold"])

            # Create Partial mask to be appended in the Allmasks List
            partialMask=np.zeros((frame.shape[0],frame.shape[1]))
            partialMask[startY:endY,startX:endX]=mask.astype(int)
            AllMasks.append(partialMask)
            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # And make it transparent by blending the color with the ROI
            color = ClassColors[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
            
            # Get the center of the object 
            ObjectsCenters.append([(startY+endY)/2,(startX+endX)/2])
            # draw the predicted label and associated probability of
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Loop on all masks and them to the combined mask array
    for smallMask in AllMasks:
        CombinedMask=CombinedMask+smallMask
    
    # Calculate the distance between each two objects 
    dist =[]
    for p1 in ObjectsCenters:
        for p2 in ObjectsCenters:
            if not (p1 == p2):
                dist.append(math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2))
                
    # Add a top black margin to the frame and put the text in it             
    frameWithBorder = cv2.copyMakeBorder( frame, 100, 0, 0,0, cv2.BORDER_CONSTANT)
    
    # Text parameters 
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    org = (250, 80)       
    # fontScale 
    fontScale = 2      
    # Line thickness of 2 px 
    thickness = 4

    # Set the minimum acceptable distance (slowingThreshold) to a value that is suitable to your video resolution and size
    #then check if any two objects in the video is less than this threshold but not collided yet
    # Display the warning SLOW!!!!
    slowingThreshold = 150
    if (len([*filter(lambda x: x < slowingThreshold, dist)]) > 0) and ((2 in CombinedMask) == False) :
        cv2.putText(frameWithBorder, 'SLOW!!', org, font,  
                   fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
        NoticeFrame+=1
        if NoticeFrame==1:
            NoticeTime = str(int(j*skipFactor*1000/30))
    if(NoticeFrame>=1):
        
        cv2.putText(frameWithBorder, 'noticed after:'+ NoticeTime +' ms', (0,50), font,  0.6, (0, 255, 255), 1, cv2.LINE_AA) 
        
    # Detect collision part the idea is when appending binary masks to Combined mask if any two pixels 
    # share the same cordinates so the pixel will have the value of 2, so we check if CombinedMask has 
    # Value of 2
    if ((2 in CombinedMask) == True):
        cv2.putText(frameWithBorder, 'Collision!!', org, font,  
                   fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
        collisionFrame+=1
        if collisionFrame==1:
            CollTime = str(int(j*skipFactor*1000/30))
    if(collisionFrame>=1):
        cv2.putText(frameWithBorder, 'crashed after:'+ CollTime +' ms', (0,80), font,  0.6, (0, 255, 255), 1, cv2.LINE_AA) 
       # Uncomment the next line and put any desired path, it will save collision frames
#        cv2.imwrite("C:/Zeerak/Desktop",frameWithBorder)
        
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, skipFactor,
                                 (frameWithBorder.shape[1], frameWithBorder.shape[0]), True)

    # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total/skipFactor))
    
    # write the output frame to disk
    writer.write(frameWithBorder)
    
    #increment the frame counter
    j+=1
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
