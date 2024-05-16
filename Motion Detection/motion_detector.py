import cv2
import time
import numpy as np
import random
import argparse

class MotionDetector:
    def __init__(self, source=0, alpha=0.15, fps=30, threshold=30, min_contour_area=1000, blur_kernel_size=(15, 15), recording_fps=10.0):
        """
        Initializes MotionDetector with customizable parameters.

        Args:
            source (int, optional): Video source. Defaults to 0.
            alpha (float, optional): Alpha value for frame averaging. Defaults to 0.15.
            fps (int, optional): Frames per second. Defaults to 30.
            threshold (int, optional): Threshold for frame difference. Defaults to 30.
            min_contour_area (int, optional): Minimum contour area for detection. Defaults to 1000.
            blur_kernel_size (tuple, optional): Kernel size for Gaussian blur. Defaults to (15, 15).
            recording_fps (float, optional): Frames per second for recording. Defaults to 10.0.
        """
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.alpha = alpha
        self.prev_frame = None
        self.recording = False
        self.out = None
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.blur_kernel_size = blur_kernel_size
        self.recording_fps = recording_fps

    def find_convex_hull_center(self, contour):
        """
        Finds the center of the convex hull of a contour.

        Args:
            contour: Contour points.

        Returns:
            tuple: (cx, cy) coordinates of the center.
        """
        hull = cv2.convexHull(contour)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        else:
            return None

    def run_detection(self):
        """
        Runs motion detection.
        """
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        display_counter = 0
        display_limit = 0
        while True:
            ret, frame = self.cap.read()
            st_tm = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)

            if self.prev_frame is None:
                self.prev_frame = gray.astype(float)
                continue

            cv2.accumulateWeighted(gray, self.prev_frame, self.alpha)
            avg_frame = cv2.convertScaleAbs(self.prev_frame)

            frame_diff = cv2.absdiff(gray, avg_frame)
            _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_detected_points = []
            miny_topl = []
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    all_detected_points.extend(contour.reshape(-1, 2))
                    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
                    min_y = [x[0][1] for x in contour]
                    for i in contour:
                        miny_topl.append(i[0])
                    display_counter = display_limit
            miny = sorted(miny_topl, key=lambda x: x[1])[0] if len(miny_topl) > 0 else [0, 0]

            if all_detected_points:
                all_detected_points = np.array(all_detected_points)
                hull = cv2.convexHull(all_detected_points)
                coords = self.find_convex_hull_center(hull)
                cx, cy = coords if coords is not None else (0, 0)
                cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), radius=15, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, tuple(miny), radius=7, color=(255, 0, 0), thickness=-1)
            cv2.imshow('Motion Detection', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and not self.recording:
                print("Recording Started")
                self.out = cv2.VideoWriter(f'output_{random.randint(1, 9999)}.mp4', fourcc, self.recording_fps, (640, 480))
                self.recording = True
            elif key == ord('s') and self.recording:
                print("Recording Stopped")
                self.out.release()
                self.recording = False
            if self.recording:
                self.out.write(frame)
            elif key == ord('q'):
                break

        self.cap.release()
        if self.recording:
            self.out.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Detection Script")
    parser.add_argument("--source", type=int, default=5, help="Source index for the video stream (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.15, help="Weight of the current frame when accumulating the average frame (default: 0.15)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the video stream (default: 30)")
    parser.add_argument("--threshold", type=int, default=30, help="Threshold value for detecting motion (default: 30)")
    parser.add_argument("--min_contour_area", type=int, default=1000, help="Minimum contour area to be considered as motion (default: 1000)")
    parser.add_argument("--blur_kernel_size", type=str, default="15,15", help="Size of the kernel for Gaussian blur in the format 'height,width' (default: '15,15')")
    parser.add_argument("--recording_fps", type=float, default=10.0, help="Frames per second for recording motion (default: 10.0)")
    args = parser.parse_args()


    detector = MotionDetector(
        source=args.source,
        alpha=args.alpha,
        fps=args.fps,
        threshold=args.threshold,
        min_contour_area=args.min_contour_area,
        blur_kernel_size=args.blur_kernel_size,
        recording_fps=args.recording_fps
    )

    # Run the motion detection
    detector.run_detection()

