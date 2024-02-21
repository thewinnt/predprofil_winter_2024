import cv2
import numpy as np

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

success, frame = cam.read()

def nothing(x):
    print(lh)

cv2.namedWindow("mask")

#(37, 43, 69) (185, 189, 231)

cv2.createTrackbar('lh', "mask", 37, 255, nothing)
cv2.createTrackbar('ls', "mask", 43, 255, nothing)
cv2.createTrackbar('lv', "mask", 69, 255, nothing)
cv2.createTrackbar('hh', "mask", 185, 255, nothing)
cv2.createTrackbar('hs', "mask", 189, 255, nothing)
cv2.createTrackbar('hv', "mask", 231, 255, nothing)

while (True):
    #read
    success, frame_ = cam.read()
    
    frame = cv2.resize(frame_, (960, 540))
    
    #process
    hsv  = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    lh = cv2.getTrackbarPos('lh', "mask")
    ls = cv2.getTrackbarPos('ls', "mask")
    lv = cv2.getTrackbarPos('lv', "mask")
    hh = cv2.getTrackbarPos('hh', "mask")
    hs = cv2.getTrackbarPos('hs', "mask")
    hv = cv2.getTrackbarPos('hv', "mask")
    
    lb = (lh, ls, lv)
    hb = (hh, hs, hv)
    
    #print(lb, hb)
    
    mask = cv2.inRange(hsv, lb, hb)
    
    #ksz = 3
    #kernel = np.ones((ksz, ksz), np.uint8)
    
    clear = mask#cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(clear, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    #centroids = output[3]
    
    for i in range(num_labels):
        if (stats[i, cv2.CC_STAT_AREA] < 10000):
            clear[np.where(labels == i)] = 0
    
    print("")
    
    #output
    #cv2.imshow("frame", frame)
    #cv2.imshow("hsv", hsv)
    
    concat = np.concatenate((mask, clear), axis=0)
    
    frame[:, :, 1] += clear // 3
    cv2.imshow("highlighted", frame)
    cv2.imshow("mask", concat)
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