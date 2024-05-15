import cv2
import numpy as np
import argparse

class MotionValidator:
    def __init__(self, compartment_points, motion_threshold=0.75):
        """
        Analyzes motion within compartments and determines its validity.

        Args:
            compartment_points (list): Points defining the compartment.
            motion_threshold (float, optional): Threshold for motion validity. Defaults to 0.75.
        """
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                poly_n=5, poly_sigma=1.2, flags=0)
        self.compartment_points = np.array(compartment_points)
        self.motion_threshold = motion_threshold

    def analyze_motion(self, start_frame, end_frame) -> str:
        """
        Analyzes motion between start and end frames.

        Args:
            start_frame (numpy.ndarray): Start frame.
            end_frame (numpy.ndarray): End frame.

        Returns:
            str: Status indicating motion validity.
        """
        start_gray = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
        end_gray = cv2.cvtColor(end_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow between start and end frames
        flow = cv2.calcOpticalFlowFarneback(start_gray, end_gray, None, **self.flow_params)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask = np.zeros_like(start_gray, dtype=np.uint8)
        cv2.fillPoly(mask, [self.compartment_points], 255)

        compartment_flow_mag = cv2.bitwise_and(magnitude, magnitude, mask=mask)
        avg_flow_magnitude = np.sum(compartment_flow_mag) / cv2.countNonZero(mask)

        return "valid" if avg_flow_magnitude > self.motion_threshold else "invalid"





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Motion Validator Script")
    parser.add_argument("image1", type=str, help="Path to the first image file")
    parser.add_argument("image2", type=str, help="Path to the second image file")
    parser.add_argument("--compartment_points", nargs='+', type=int, default=[[190, 3], [148, 393], [544, 401], [561, 4]],
                        help="Coordinates for 640x480 image from the same camera, converted to a list of lists")
    args = parser.parse_args()


    # Coords for 640x480 image from the same camera, converted to a list of lists
    compartment_points = args.compartment_points
    mv = MotionValidator(compartment_points)

    # Resizing images to match 640x480
    img1 = cv2.resize(cv2.imread(args.image1), (640, 480))
    img2 = cv2.resize(cv2.imread(args.image2), (640, 480))
    status = mv.analyze_motion(img1, img2)
    print("Motion status:", status)

