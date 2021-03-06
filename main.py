import cv2
import numpy as np

from networktables import NetworkTables

NetworkTables.initialize(server='10.68.29.2')

sd = NetworkTables.getTable('Vision')


import subprocess
import time

subprocess.Popen(["mjpg_streamer", "-i", "/usr/local/lib/input_file.so -f /var/tmp/ -n pic.jpg -r","-o", "/usr/local/lib/output_http.so -w /usr/local/www -p 1180s"])
time.sleep(.5)

cap = cv2.VideoCapture(0)

lower_hsv = np.array([20, 71, 0])
upper_hsv = np.array([45, 255, 255])


def validate_contour(cnt, cntArea, rect):
    return True
    target = 1
    tolerance = 1

    (x, y, w, h) = rect

    rectArea = (w * h)
    
    ratio = rectArea / cntArea

    print(ratio)

    if abs(ratio - target) < tolerance:
        return True
    return False



def get_box(contours):
    result = []
    if len(contours) == 0:
        return None
    
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sortedContours = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse= True)

    hull = None
    for i in sortedContours:
        area = i[0]
        cnt = i[1]
        # epsilon = 0.02*cv2.arcLength(cnt, True)
        # hull = cv2.approxPolyDP(cnt, epsilon, True)

        hul = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(hul)

        rect = (x,y,w,h)

        if validate_contour(hul, area, rect):
            hull = hul
            break

    M = cv2.moments(hull)
    try:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        return None
    result = [hull, area, (x,y), rect]
    return result

while True:
    _, OG_img = cap.read()
    OG_img = cv2.resize(OG_img, (320,240))

    img_width = 320
    img_height = 240

# img = cv2.GaussianBlur(OG_img, (5,5), 0)
    # img = cv2.medianBlur(img, 5)
    img = OG_img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernal = np.ones((9,9), np.uint8)
    thresh = cv2.erode(thresh, kernal, iterations=1)
    thresh = cv2.medianBlur(thresh, 15)
    thresh = cv2.dilate(thresh, kernal, iterations=1)

    _, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    result = get_box(contours)

    sd.putBoolean('target_found', False)
    if result is not None:
        [hull, area, center, rect] = result 
        (x, y ,w, h ) = rect

        cv2.drawContours(OG_img, [hull], -1, (255, 0, 0), 3)
        cv2.rectangle(OG_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        radius = 2
        cv2.circle(OG_img, center, radius, (0, 0, 255), 2)
        center_x = -int(img_width/2) + center[0]
        

        sd.putNumber('x', center_x)
        sd.putNumber('y', center[1])
        sd.putNumber('area', area)
        sd.putNumber('relArea', (area / (img_width * img_height)))
        sd.putBoolean('target_found', True)

    sd.putNumber('img_width', img_width)
    sd.putNumber('img_height', img_height)
    cv2.imshow('result', OG_img)
    cv2.imshow('thresh', thresh)
# 
    driver_img = cv2.resize(OG_img, (80,60))
#     cv2.imshow("img", driver_img)
    cv2.imwrite("/../var/tmp/pic.jpg", driver_img)
    cv2.waitKey(1)
