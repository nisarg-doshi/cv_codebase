import cv2
import numpy as np
import argparse

class ObjectDetector:
    """
    Class for detecting objects in images using edge detection and contour analysis.
    """

    def __init__(self, overlap_thresh=0.5):
        self.overlap_thresh = overlap_thresh

    def non_max_suppression_fast(self, boxes):
        """
        Apply non-maximum suppression to eliminate redundant bounding boxes.

        Args:
            boxes (numpy.ndarray): Array of bounding boxes.

        Returns:
            numpy.ndarray: Array of picked bounding boxes.
        """
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.overlap_thresh)[0])))

        return boxes[pick].astype("int")

    def detect_objects(self, image):
        """
        Detect objects in the input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with detected objects.
        """
        original_image = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(blurred, kernel, iterations=1)

        edged = cv2.Canny(dilate, 75, 350)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(edged, kernel, iterations=2)

        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])

        nms_boxes = self.non_max_suppression_fast(np.array(boxes))

        for box in nms_boxes:
            x1, y1, x2, y2 = box
            original_image = cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            original_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Detected Objects", original_image)
        cv2.waitKey(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection in images using edge detection and contour analysis.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    detector = ObjectDetector()
    image = cv2.imread(args.image_path)
    detector.detect_objects(image)
