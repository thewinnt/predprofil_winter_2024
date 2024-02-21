import numpy as np
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from ultralytics.engine.results import Results, Boxes

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1)

video = cv2.VideoWriter("output.avi", cv2.VideoWriter.fourcc(*"MJPG"), 5, (960, 540))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) #cap.set(3, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) #cap.set(4, 480)

print(model.names)

while True:
    _, img = cap.read()
    _, img = cap.read()

    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results: list[Results] = model.predict(img)

    for r in results:

        annotator = Annotator(img)

        boxes: Boxes = r.boxes # type: ignore
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            if model.names[int(c)] != "person":
                continue
            annotator.box_label(b, f"{model.names[int(c)]} {box.conf.tolist()[0]:.3f}")

    img = annotator.result()
    video.write(img.astype(np.uint8))
    cv2.imshow('YOLO V8 Detection', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()