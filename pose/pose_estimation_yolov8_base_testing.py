import cv2
import traceback
import numpy as np
from ultralytics import YOLO
import random

def euc_dis(cord1, cord2):
    """
    Calculates the Euclidean distance between two points.

    Args:
    - cord1 (tuple): Coordinates of the first point.
    - cord2 (tuple): Coordinates of the second point.

    Returns:
    - distance (float): Euclidean distance between the points.
    """
    point1 = np.array(cord1)
    point2 = np.array(cord2)
    sum_sq = np.sum(np.square(point1 - point2))
    return np.sqrt(sum_sq)

def inference_live(frame_d):
    """
    Performs live inference using YOLOv8 model on a frame and draws all key pose points.

    Args:
    - frame_d (numpy.ndarray): Input frame.

    Returns:
    - dum_fr (numpy.ndarray): Frame with inferred keypoints and confidence values.
    """
    in_id = random.randint(1, 999999)
    result = model.track(frame_d, conf=0.25, verbose=False)
    try:
        model_result_conf = list(result[0].keypoints.conf.cpu().tolist())
        model_result_kp = list(result[0].keypoints.xy.cpu().tolist())
        for confs, kps in zip(model_result_conf, model_result_kp):
            for kp in kps:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame_d, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    except:
        return frame_d
    
    return frame_d

def main():
    # Initialize YOLO model
    model = YOLO('yolov8s-pose.pt')

    # Initialize video capture
    cam = cv2.VideoCapture(3)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

    while True:
        ret, frame = cam.read()
        cv2.imshow("Result Frame", inference_live(frame))
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
    main()
