import cv2
import os
import numpy as np

# YOLO Versions
YOLO_V5 = 5
YOLO_V8 = 8

# Constants
_INPUT_WIDTH = 640
_INPUT_HEIGHT = 640
_SCORE_THRESHOLD = 0.2
_NMS_THRESHOLD = 0.4
_CONFIDENCE_THRESHOLD = 0.4

class YoloDetector:

    def __init__(self, version = YOLO_V5, is_cuda = False, onnx_file_path = None, class_list_file_path = None):
        self._version = version

        if onnx_file_path is None:
            if (version == YOLO_V5):
                onnx_file_path = os.path.join(os.path.dirname(__file__), "models/YOLOv5s.onnx")
            elif (version == YOLO_V8):
                onnx_file_path = os.path.join(os.path.dirname(__file__), "models/YOLOv8s.onnx")

        if class_list_file_path is None:
            class_list_file_path = os.path.join(os.path.dirname(__file__), "models/classes.txt")

        self._build_model(onnx_file_path, is_cuda)

        self._class_list = []
        with open(class_list_file_path, "r") as classes_file:
            self._class_list = [class_name.strip() for class_name in classes_file.readlines()]

    def apply(self, image):
        yolo_image = self._format_yolo(image)
        detections = self._detect(yolo_image)
        return self._wrap_detection(yolo_image, detections[0])

    def _build_model(self, onnx_file_path, is_cuda):
        self._net = cv2.dnn.readNet(onnx_file_path)

        if is_cuda:
            print("Attempty to use CUDA")
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (_INPUT_WIDTH, _INPUT_HEIGHT), swapRB = True, crop = False)
        self._net.setInput(blob)
        output = self._net.forward()
        # output = net.forward(net.getUnconnectedOutLayersNames())

        if self._version == YOLO_V8:
            output = output.transpose((0, 2, 1))

        return output

    def _wrap_detection(self, image, detections):
        class_ids = []
        confidences = []
        boxes = []

        rows = detections.shape[0]

        image_width, image_height, _ = image.shape

        x_factor = image_width / _INPUT_WIDTH
        y_factor =  image_height / _INPUT_HEIGHT

        for r in range(rows):
            row = detections[r]
            confidence = row[4]

            if (self._version == YOLO_V5 and confidence < _CONFIDENCE_THRESHOLD):
                continue

            if (self._version == YOLO_V5):
                classes_scores = row[5:]
            elif (self._version == YOLO_V8):
                classes_scores = row[4:]

            _, _, _, max_index = cv2.minMaxLoc(classes_scores)
            class_id = max_index[1]

            if (classes_scores[class_id] >= _SCORE_THRESHOLD):
                if (self._version == YOLO_V5):
                    confidences.append(confidence)
                elif (self._version == YOLO_V8):
                    confidences.append(classes_scores[class_id])

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, _CONFIDENCE_THRESHOLD, _NMS_THRESHOLD)

        result_class_ids = []
        result_class_names = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_class_ids.append(class_ids[i])
            result_class_names.append(self._class_list[class_ids[i]])
            result_confidences.append(confidences[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_class_names, result_confidences, result_boxes

    def _format_yolo(self, image):
        row, col, _ = image.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = image
        return result
