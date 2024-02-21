import cv2
import numpy as np

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# success, frame = cam.read()
frame = cv2.imread("1.jpg")

def nothing(x):
    pass

cv2.namedWindow("mask")

#(37, 43, 69) (185, 189, 231)

cv2.createTrackbar('lh', "mask", 0, 255, nothing)
cv2.createTrackbar('ls', "mask", 0, 255, nothing)
cv2.createTrackbar('lv', "mask", 0, 255, nothing)
cv2.createTrackbar('hh', "mask", 255, 255, nothing)
cv2.createTrackbar('hs', "mask", 255, 255, nothing)
cv2.createTrackbar('hv', "mask", 255, 255, nothing)
cv2.createTrackbar('layer', "mask", 1, 10, nothing)

while (True):
    #read
    # success, frame = cam.read()
    
    # frame = cv2.resize(frame_, (960, 540))
    
    #process
    hsv  = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    lh = cv2.getTrackbarPos('lh', "mask")
    ls = cv2.getTrackbarPos('ls', "mask")
    lv = cv2.getTrackbarPos('lv', "mask")
    hh = cv2.getTrackbarPos('hh', "mask")
    hs = cv2.getTrackbarPos('hs', "mask")
    hv = cv2.getTrackbarPos('hv', "mask")
    layer = cv2.getTrackbarPos('layer', "mask")
    
    lb = np.array((lh, ls, lv), np.uint8)
    hb = np.array((hh, hs, hv), np.uint8)
    
    #print(lb, hb)
    
    mask = cv2.inRange(hsv, lb, hb)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # frame[:, :, 1] += clear // 3
    frame2 = frame.copy()
    cv2.drawContours(frame2, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, layer)

    cv2.imshow("highlighted", frame2)
    cv2.imshow("mask", mask)
    #cv2.imshow("clear", clear)
    #cv2.imshow("h", hsv[:, :, 0])
    #cv2.imshow("s", hsv[:, :, 1])
    #cv2.imshow("v", hsv[:, :, 2])
    
    #process keyboard
    key = cv2.waitKey(200) & 0xFF
    
    if (key == ord('q')):
        break

cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
cv2.waitKey(10)