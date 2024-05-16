import cv2
import numpy as np
import random
import argparse
from ultralytics import YOLO

class PoseEstimator:
    """
    Class for performing live pose estimation using YOLOv8 model on video frames.
    """

    def __init__(self, model_path='yolov8s-pose.pt'):
        """
        Initializes the PoseEstimator object.

        Args:
            model_path (str): Path to the YOLOv8 model weights.
        """
        self.model = YOLO(model_path)

    def inference_live(self, frame):
        """
        Performs live inference using YOLOv8 model on a frame and draws all key pose points.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            numpy.ndarray: Frame with inferred keypoints and confidence values.
        """
        result = self.model.track(frame, conf=0.25, verbose=False)
        try:
            model_result_conf = list(result[0].keypoints.conf.cpu().tolist())
            model_result_kp = list(result[0].keypoints.xy.cpu().tolist())
            for confs, kps in zip(model_result_conf, model_result_kp):
                for kp in kps:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        except:
            pass
        return frame

def main(model_path='yolov8s-pose.pt'):
    """
    Main function for live pose estimation using YOLOv8 model on video frames.

    Args:
        model_path (str): Path to the YOLOv8 model weights.
    """
    pose_estimator = PoseEstimator(model_path)

    # Initialize video capture
    cam = cv2.VideoCapture(3)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

    while True:
        ret, frame = cam.read()
        cv2.imshow("Result Frame", pose_estimator.inference_live(frame))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.imwrite('captured_image.png', frame)
            print('Image captured!')
        elif key == ord('q'):
            break

    # Release video capture and close windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live pose estimation using YOLOv8 model.")
    parser.add_argument("--model_path", type=str, default='yolov8s-pose.pt', help="Path to the YOLOv8 model weights.")
    args = parser.parse_args()
    main(args.model_path)
