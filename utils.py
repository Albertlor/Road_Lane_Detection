import cv2
import numpy as np
import warnings


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def empty(a):
    pass


def canny(img, thresh1, thresh2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, thresh1, thresh2)
    return edged


def cannyThresh_trackbar():
    cv2.namedWindow("Edge Threshold")
    cv2.resizeWindow("Edge Threshold", 360, 100)
    cv2.createTrackbar("Threshold 1", "Edge Threshold", 90, 500, empty)
    cv2.createTrackbar("Threshold 2", "Edge Threshold", 180, 500, empty)


def val_CannyThresh():
    thresh1 = cv2.getTrackbarPos("Threshold 1", "Edge Threshold")
    thresh2 = cv2.getTrackbarPos("Threshold 2", "Edge Threshold")
    return thresh1, thresh2


def warpImg(img, points, w, h):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h))
    return M, warped


def pointsToWarp_trackbar(wT=480, hT=240):
    cv2.namedWindow("Points to Warp")
    cv2.resizeWindow("Points to Warp", 360, 180)
    cv2.createTrackbar("Width Top", "Points to Warp", 166, wT*2, empty)
    cv2.createTrackbar("Height Top", "Points to Warp", 74, hT, empty)
    cv2.createTrackbar("Width Bottom", "Points to Warp", 0, wT*2, empty)
    cv2.createTrackbar("Height Bottom", "Points to Warp", 240, hT, empty)


def val_pointsToWarp(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Points to Warp")
    heightTop = cv2.getTrackbarPos("Height Top", "Points to Warp")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Points to Warp")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Points to Warp")

    points = np.float32([[widthTop + 40, heightTop + 200], [wT - widthTop + 200, heightTop + 200],
                         [widthBottom + 40, heightBottom + 200], [wT - widthBottom + 200, heightBottom + 200]])
    return points


def drawWarpedPoints(img, points):
    for i in range(4):
        cv2.circle(img, (int(points[i][0]), int(points[i][1])), 10, (255, 0, 0), -1)
    return img


def houghThresh_trackbar():
    cv2.namedWindow("Hough Parameters")
    cv2.resizeWindow("Hough Parameters", 360, 150)
    cv2.createTrackbar("Hough Threshold", "Hough Parameters", 54, 1000, empty)
    cv2.createTrackbar("Max Line Gap", "Hough Parameters", 10, 500, empty)
    cv2.createTrackbar("Min Line Length", "Hough Parameters", 0, 500, empty)


def val_houghThresh():
    houghThresh = cv2.getTrackbarPos("Hough Threshold", "Hough Parameters")
    maxLineGap = cv2.getTrackbarPos("Max Line Gap", "Hough Parameters")
    minLineLength = cv2.getTrackbarPos("Min Line Length", "Hough Parameters")

    params = np.array([houghThresh, maxLineGap, minLineLength])
    return params


# def restoreImg(img, points, w, h):
#     pts1 = np.float32(points)
#     pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     restored = cv2.warpPerspective(img, M, (w, h))
#     return M, restored


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

