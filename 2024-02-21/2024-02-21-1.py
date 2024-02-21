import cv2

video = cv2.VideoCapture(1)  # i think cam 1 is obs?
# video.set(cv2.CAP_PROP_FRAME_WIDTH, 1536)
# video.set(cv2.CAP_PROP_FRAME_HEIGHT, 864)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, frame = video.read()

    if success:
        frame = frame[:, :, ::-1]
        # cv2.applyColorMap(frame, cv2.COLORMAP_HSV, frame)

        cv2.imshow("Camera Capture", frame)
        key = cv2.waitKey(1)  # waits 1 ms for a key press and closes the window, returning the key
        if key == ord(' '):
            break

print(frame.shape)

video.release()
cv2.destroyAllWindows()