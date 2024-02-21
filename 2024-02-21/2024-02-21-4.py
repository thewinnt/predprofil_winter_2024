import time
import cv2
import numpy as np

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time1 = time.process_time_ns()
time2 = time.process_time_ns()

while (True):
    #read
    success, frame = cam.read()

    # convert to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))
    image = cv2.filter2D(grayscale, -1, kernel)

    time2 = time.process_time_ns()
    print(f"FPS: {1 / ((time2 - time1) * 1e9 + .00001)}")
    time1 = time2
    
    cv2.imshow("frame", image)
    
    #process keyboard
    key = cv2.waitKey(10)
    if (key == ord('q')):
        break

cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
cv2.waitKey(10)