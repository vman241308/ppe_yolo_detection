import cv2
import time
import sys
from YOLO_detector import *

COLORS = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

# Main function
if __name__== "__main__":

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    # yolo_detector = YoloDetector(YOLO_V5, is_cuda)
    yolo_detector = YoloDetector(YOLO_V8, is_cuda)

    capture = cv2.VideoCapture("../Test_Video_Files/PPE.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Initialize calculating FPS
    start = time.time_ns()
    frame_count = 0
    fps = -1

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        class_ids, class_names, confidences, boxes = yolo_detector.apply(frame)

        frame_count += 1

        for (class_id, class_name, confidence, box) in zip(class_ids, class_names, confidences, boxes):
            color = COLORS[int(class_id) % len(COLORS)]
            # label = "%s (%d%%)" % (class_name, int(confidence * 100))
            label = "%s" % (class_name)

            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        # if fps > 0:
        #     fps_label = "FPS: %.2f" % fps
        #     cv2.putText(frame, fps_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
