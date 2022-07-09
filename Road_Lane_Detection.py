#python Road_Lane_Detection.py -v video/road.mp4
#python Road_Lane_Detection.py -v video/road1.mp4
#python Road_Lane_Detection.py -i image/road.jpg
#python Road_Lane_Detection.py -v Udacity_Advanced_Lane_Line_Detection_P4/project_video.mp4


import argparse
import cv2
import numpy as np
import imutils
import time
import warnings
import utils

cache_left = []
cache_right = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to the input image")
    args = vars(ap.parse_args())

    # create edge detection threshold trackbars
    utils.cannyThresh_trackbar()

    # create points to warp trackbars
    utils.pointsToWarp_trackbar()

    # create hough transform parameters trackbars
    utils.houghThresh_trackbar()

    video = cv2.VideoCapture(args["video"])
    count = 0
    last_time = time.time()

    while True:
        count += 1
        if video.get(cv2.CAP_PROP_FRAME_COUNT) == count:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0

        ret, frame = video.read()
        duration = time.time() - last_time
        last_time = time.time()
        fps = str(round((1/duration), 2))
        if ret:
            frame = imutils.resize(frame, width=800, inter=cv2.INTER_LINEAR)
            cv2.putText(frame, "fps: " + fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            ###STEP 1, Edge detection
            thresh = utils.val_CannyThresh()
            thresh1, thresh2 = thresh
            edged = utils.canny(frame, thresh1, thresh2)

            ###STEP 2, Crop region of interest
            h, w = frame.shape[:2]
            triangle = np.array([(300, 300), (0, h), (w, h)])
            mask = np.zeros_like(edged)
            cv2.fillConvexPoly(mask, triangle, (255, 255, 255), cv2.LINE_AA)
            roi = cv2.bitwise_and(edged, edged, mask=mask)

            ###STEP 3, Transform to bird-eye view
            h, w = frame.shape[:2]
            roi_copy = roi.copy()
            points = utils.val_pointsToWarp()
            M, warped = utils.warpImg(roi_copy, points, w, h)
            roi_copy2 = roi.copy()
            roi_copy2 = cv2.cvtColor(roi_copy2, cv2.COLOR_GRAY2BGR)
            warpedPoints = utils.drawWarpedPoints(roi_copy2, points)

            ##STEP 4, Detection lines using Hough transform
            warped_copy = warped.copy()
            warped_copy = cv2.cvtColor(warped_copy, cv2.COLOR_GRAY2BGR)
            houghThresh, maxLineGap, minLineLength = utils.val_houghThresh()
            lines = cv2.HoughLinesP(warped, 1, np.pi / 180, houghThresh, minLineLength=minLineLength, maxLineGap=maxLineGap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(warped_copy, (x1, y1), (x2, y2), (255, 0, 0), 5)

            ###STEP 7, Reverse the bird-eye view
            invM, restored = utils.restoreImg(warped_copy, points, w, h)
            final = cv2.addWeighted(frame, 0.6, restored, 1, 0)

            #display
            screen = utils.stackImages(0.5, ([frame, edged, roi], [warpedPoints, warped, final]))
            cv2.imshow("Frames", screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()