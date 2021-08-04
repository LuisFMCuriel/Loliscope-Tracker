from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import sys
from tifffile import imread, imsave
import imutils


img_dir = r"X:\LuisFel\Tracking\Fake_neurons_track"
path_s = r"X:\LuisFel\Tracking\Neuron_Autotracked"
data_path = os.path.join(img_dir, "*.tif") #Assume images are in tiff format
img_files = glob.glob(data_path)

tracked = np.zeros((100,500,500,3))

# Read first image
frame = imread(img_files[0])
original = np.copy(frame)
frame = ((frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)))*255.0
frame = frame.astype("uint8")

gray  = frame
thresh = cv2.threshold(gray, 80, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
im = gray.copy()   
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=3,  
    labels=thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
boxes_coord = []
boxes_pix = []
conts=[]    
avg = []
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    boxes_pix.append(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box) 
    if cv2.contourArea(c) > 40:
        mask = np.copy(frame)
        boxes_coord.append(box)
        #cv2.drawContours(image,c,-1,(0,255,0))
        #cv2.drawContours(image,[box],-1,(255,255,255))
        cv2.drawContours(mask,[box],-1,(255,255,255), -1)
        #res = cv2.bitwise_and(original,original,mask=mask)
        idx = np.where( mask==255)
        avg.append(np.mean(original[idx]))

frame = imutils.resize(frame, width=500)

#For now only track one neuron
colors = []
bboxes = []
pixels_frame = 10
for box in boxes_coord:
	xmin = np.amin(box[:,0])
	ymin = np.amin(box[:,1])
	xmax = np.amax(box[:,0])
	ymax = np.amax(box[:,1])
	w = ymax - ymin
	h = xmax - xmin
	#Normalize the coordinates for the new size
	xmin = int((frame.shape[0]*(xmin))/original.shape[0]) - pixels_frame
	ymin = int((frame.shape[0]*(ymin))/original.shape[0]) - pixels_frame
	bboxes.append((xmin,ymin,h,w))
	colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

#h = int((frame.shape[0]*(h))/original.shape[0]) + pixels_frame
#w = int((frame.shape[0]*(w))/original.shape[0]) + pixels_frame


if frame is None:
    print("Cannot read image file")
    sys.exit()

# Set the coord of the first neuron
#bbox = (xmin, ymin, h, w)
#print(bbox)
#bbox = cv2.selectROI(frame, False)
#print(bbox)
#tracker = cv2.legacy.TrackerMOSSE_create()
#ok = tracker.init(frame, bbox)

#Create multitracker object
tracker = cv2.legacy.MultiTracker_create()

#Tracker type
trackerType = "CSRT"
#Initialize multitracker
for bbox in bboxes:
	tracker.add(cv2.legacy.TrackerMOSSE_create(), frame, bbox)


#while True:
cont = 0
# Iterate image files instead of reading from a video file
for f1 in img_files:
	#frame = cv2.imread(f1)
	frame = imread(f1)
	frame = ((frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)))*255.0
	frame = frame.astype("uint8")
	frame = np.expand_dims(frame, axis=-1)
	frame = np.broadcast_to(frame, (frame.shape[0], frame.shape[1], 3))
	frame = imutils.resize(frame, width=500)

	# Start timer
	timer = cv2.getTickCount()

	# Update tracker
	ok, bbox = tracker.update(frame)
	print(ok)

	# Calculate Frames per second (FPS)
	#fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

	fps = 8 # We don't know the fps from the set of images

	for i, newbox in enumerate(bboxes):
		p1 = (int(newbox[0]), int(newbox[1]))
		p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
		cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

	'''
    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2, 1)
        #cv2.rectangle(frame.astype("uint8"), p1, p2, (255,0,0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     '''
    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
	cv2.imshow("Tracking", frame)
	frame = ((frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)))*255.0
	tracked[cont,:,:,:] = frame
	cont += 1
	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff 

	if k == 27:
		break
imsave(os.path.join(path_s, "fake_neuron.tif"), tracked.astype("uint8"))
