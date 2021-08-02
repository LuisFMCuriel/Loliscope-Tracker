import cv2
import os
import glob
import sys
from tifffile import imread, imsave
import numpy as np
import imutils
# https://www.quora.com/How-can-I-read-multiple-images-in-Python-presented-in-a-folder
img_dir = "C:/Images"  # Enter Directory of all images 
img_dir = r"X:\LuisFel\Tracking\worm_moving_imgs"
path_s = r"X:\LuisFel\Tracking\Tracked_worm_moving_imgs"
data_path = os.path.join(img_dir, "*.tif") #Assume images are in tiff format
img_files = glob.glob(data_path)

tracked = np.zeros((100,314,500,3))

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
	tracker = cv2.legacy.TrackerMOSSE_create()
    #tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()


if not img_dir:
    print("Images folder is empty")
    sys.exit()

# Read first image
frame = cv2.imread(img_files[0])
frame = ((frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)))*255.0
frame = frame.astype("uint8")
#frame = np.expand_dims(frame, axis=-1)
#frame = np.broadcast_to(frame, (frame.shape[0], frame.shape[1], 3))
frame = imutils.resize(frame, width=500)

if frame is None:
    print("Cannot read image file")
    sys.exit()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

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

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)
    frame = ((frame - np.amin(frame))/(np.amax(frame) - np.amin(frame)))*255.0
    tracked[cont,:,:,:] = frame
    cont += 1
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff 

    if k == 27:
        break
imsave(os.path.join(path_s, "worm-tracked.tif"), tracked.astype("uint8"))
