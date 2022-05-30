#python Road_Lane_Detection.py -v video/road.mp4
#python Road_Lane_Detection.py -i image/road.jpg

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
from stackImages.stackImages import stackImages
import warnings

cache_left = []
cache_right = []


def empty(a):
    pass


def find_coordinates(frame, lines_params):
    h, w = frame.shape[:2]

    slope, intercept = lines_params
    y1 = h
    x1 = int((y1 - intercept) / slope)
    y2 = int((3/5) * y1)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope(frame, lines):
    warnings.warn("deprecated", DeprecationWarning)
    global cache_left
    global cache_right
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    try:
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)

        left_line = find_coordinates(frame, left_fit_avg)
        right_line = find_coordinates(frame, right_fit_avg)

        cache_left = left_fit
        cache_right = right_fit
        return np.array([left_line, right_line])
    except TypeError:
        left_fit_avg = np.average(cache_left, axis=0)
        right_fit_avg = np.average(cache_right, axis=0)

        left_line = find_coordinates(frame, left_fit_avg)
        right_line = find_coordinates(frame, right_fit_avg)

        return np.array([left_line, right_line])


def main():
    video = cv2.VideoCapture(args["video"])
    count = 0
    last_time = time.time()
    global refPts1
    while True:
        ret, frame = video.read()
        duration = time.time() - last_time
        last_time = time.time()
        fps = str(round((1/duration), 2))
        if ret:
            frame = imutils.resize(frame, width=800, inter=cv2.INTER_LINEAR)
            cv2.putText(frame, "fps: " + fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

            #convert image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            #edge detection
            #get the real-time position of threshold trackbar
            thresh1 = cv2.getTrackbarPos("Threshold1", "Threshold Value")
            thresh2 = cv2.getTrackbarPos("Threshold2", "Threshold Value")

            #appy canny edge detection
            edged = cv2.Canny(gray, thresh1, thresh2)
            # kernel = np.ones((5, 5))
            # imgDil = cv2.dilate(edged, kernel, iterations=1)
            # kernel = np.ones((3, 3))
            # imgErode = cv2.erode(imgDil, kernel, iterations=1)

            #crop region of interest
            h, w = frame.shape[:2]
            triangle = np.array([(100, h), (700, h), (450, 250)])
            mask = np.zeros_like(edged)
            cv2.fillConvexPoly(mask, triangle, (255, 255,255), cv2.LINE_AA)
            roi = cv2.bitwise_and(edged, edged, mask=mask)

            #apply Hough transform
            houghThresh = cv2.getTrackbarPos("Hough", "Threshold Value")
            minLength = cv2.getTrackbarPos("Min Length", "Threshold Value")
            maxGap = cv2.getTrackbarPos("Max Gap", "Threshold Value")
            lines = cv2.HoughLinesP(roi, 2, np.pi/180, houghThresh, minLineLength=minLength, maxLineGap=maxGap)
            result = frame.copy()
            result_avg = frame.copy()

            if lines is not None:
                #draw all lines detected
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)

                #draw average lines
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    avg_lines = average_slope(result_avg, lines)
                left_line = avg_lines[0]
                right_line = avg_lines[1]
                cv2.line(result_avg, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 5, cv2.LINE_AA)
                cv2.line(result_avg, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 5, cv2.LINE_AA)

            #display
            stacked = stackImages(0.5, ([frame, gray, edged], [roi, result, result_avg]))
            cv2.imshow("Frames", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('r'):
            main()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to the input image")
args = vars(ap.parse_args())

#create a hsv window and trackbars
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE MIN", "HSV", 20, 179, empty)
cv2.createTrackbar("HUE MAX", "HSV", 30, 179, empty)
cv2.createTrackbar("SAT MIN", "HSV", 100, 255, empty)
cv2.createTrackbar("SAT MAX", "HSV", 255, 255, empty)
cv2.createTrackbar("VAL MIN", "HSV", 100, 255, empty)
cv2.createTrackbar("VAL MAX", "HSV", 255, 255, empty)

#create a threshold value window and trackbars
cv2.namedWindow("Threshold Value")
cv2.resizeWindow("Threshold Value", 640, 240)
cv2.createTrackbar("Threshold1", "Threshold Value", 50, 255, empty)
cv2.createTrackbar("Threshold2", "Threshold Value", 150, 255, empty)
cv2.createTrackbar("Hough", "Threshold Value", 54, 1000, empty)
cv2.createTrackbar("Min Length", "Threshold Value", 0, 500, empty)
cv2.createTrackbar("Max Gap", "Threshold Value", 10, 500, empty)



if __name__ == "__main__":
    main()