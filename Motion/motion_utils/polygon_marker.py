import cv2
import numpy as np
import argparse
class PolygonMarker:
    def __init__(self, image):
        """
        Initializes PolygonMarker with an image for marking polygons.

        Args:
            image (str or numpy.ndarray): Image file path or numpy array.
        """
        if isinstance(image, str):
            self.frame = cv2.imread(image)
        else:
            self.frame = image

        self.compartment1_points = []
        self.compartment2_points = []
        self.current_compartment = None

        # Set mouse callback
        cv2.namedWindow("Mark Points")
        cv2.setMouseCallback("Mark Points", self.mark_point)

    def mark_point(self, event, x, y, flags, param):
        """
        Marks points on the image when the mouse is clicked.

        Args:
            event (int): Mouse event.
            x (int): X-coordinate of the mouse click.
            y (int): Y-coordinate of the mouse click.
            flags (int): Flags for mouse event.
            param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_compartment == 1:
                self.compartment1_points.append((x, y))
            elif self.current_compartment == 2:
                self.compartment2_points.append((x, y))
            self.update_visual_feedback()

    def update_visual_feedback(self):
        """
        Updates visual feedback on the image.
        """
        feedback_frame = self.frame.copy()
        for point in self.compartment1_points:
            cv2.circle(feedback_frame, point, 5, (0, 0, 255), -1)
        for point in self.compartment2_points:
            cv2.circle(feedback_frame, point, 5, (0, 255, 0), -1)
        if len(self.compartment1_points) > 1:
            cv2.polylines(feedback_frame, [np.array(self.compartment1_points)], isClosed=True, color=(0, 0, 255), thickness=2)
        if len(self.compartment2_points) > 1:
            cv2.polylines(feedback_frame, [np.array(self.compartment2_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Mark Points", feedback_frame)

    def mark_polygons(self):
        """
        Marks polygons on the image.
        """
        while True:
            self.update_visual_feedback()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                self.current_compartment = 1
            elif key == ord('2'):
                self.current_compartment = 2
            elif key == ord('s'):
                print("Saved Points for Compartment 1:", self.compartment1_points)
                print("Saved Points for Compartment 2:", self.compartment2_points)
                break
            elif key == ord('q'):
                return "Broken prematurely", "Error"
        cv2.destroyAllWindows()
        return self.compartment1_points, self.compartment2_points



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polygon Marker Script")
    parser.add_argument("--image_file", type=str, help="Path to the image file")
    args = parser.parse_args()
    marker = PolygonMarker(args.image_file)
    
    # Mark polygons on the image
    compartment1_points, compartment2_points = marker.mark_polygons()



