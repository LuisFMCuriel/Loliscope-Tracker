# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
import ffmpeg



def Tracker(path, filename, initBB = None):
    tracker = cv2.TrackerKCF_create()
    os.system(r"C:\Users\lmorales-curiel\Documents\fiji-win64\Fiji.app\ImageJ-win64.exe --headless --console -macro ./Macro_fiji_saveAVI")

    img = os.path.join(path,filename)
    #Read video
    vs = cv2.VideoCapture(img+".avi")

    vs.set(cv2.CAP_PROP_FPS, 1)

    s = []
    cont = 0
    fps = 0
    
    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        
        frame = frame[1] #if args.get("video", False) else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            if cont == 0:
                tracker.init(frame, initBB)
                fps = FPS().start()
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            print(success, box)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame 
            info = [
                ("Tracker", "KCF"),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        
        cv2.imshow("Frame", frame)
        print(cont)
        if cont == 0 and initBB == None:
            key = 255
            while key == 255:
                key = cv2.waitKey(1000) & 0xFF
        else:
            key = cv2.waitKey(10) & 0xFF
        cont += 1
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            print(type(initBB))
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            cont = cont + 1
            fps = FPS().start()
    cv2.destroyAllWindows()
    return box

#path = r"C:\Users\lmorales-curiel\Desktop"
#name = "a"
#ROI = (13, 234, 274, 198)
#x,y,w,h = Tracker(path, name, initBB = ROI)
#print(x,y)
#img = imread(os.path.join(path, name+".tif"))
#vidwrite("a", img, framerate=1)
#os.system(r"C:\Users\lmorales-curiel\Documents\fiji-win64\Fiji.app\ImageJ-win64.exe --headless --console -macro ./Macro_fiji_saveAVI")
#os.system(r"C:\Users\lmorales-curiel\Documents\fiji-win64\Fiji.app\ImageJ-win64.exe -eval print('Hello, world');")

