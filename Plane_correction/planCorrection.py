import argparse
import cv2
import numpy as np
import math

class PerspectiveCorrection:
    """
    Class for correcting perspective in an image.
    """

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            x1 (float): x-coordinate of the first point.
            y1 (float): y-coordinate of the first point.
            x2 (float): x-coordinate of the second point.
            y2 (float): y-coordinate of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    @staticmethod
    def correct_perspective(image, points):
        """
        Corrects the perspective of an image using given points.

        Args:
            image (numpy.ndarray): Input image.
            points (list): List of four corner points specifying the region of interest.

        Returns:
            numpy.ndarray: Warped image with corrected perspective.
        """
        # Assuming points are ordered as [top-left, top-right, bottom-right, bottom-left]
        tl, tr, br, bl = points

        # Compute width as average of top and bottom widths
        top_width = PerspectiveCorrection.calculate_distance(tl[0], tl[1], tr[0], tr[1])
        bottom_width = PerspectiveCorrection.calculate_distance(bl[0], bl[1], br[0], br[1])
        width = int((top_width + bottom_width) / 2)

        # Compute height as average of left and right heights
        left_height = PerspectiveCorrection.calculate_distance(tl[0], tl[1], bl[0], bl[1])
        right_height = PerspectiveCorrection.calculate_distance(tr[0], tr[1], br[0], br[1])
        height = int((left_height + right_height) / 2)

        # Define the target points
        targets = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)
        corners = np.array(points, dtype=np.float32)

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(corners, targets)
        warped_image = cv2.warpPerspective(image, M, (width, height))

        return warped_image
    @staticmethod
    def parse_coordinates(coord_str):
        """
        Parse a string of coordinates into a list of tuples.

        Args:
            coord_str (str): String containing coordinates in the format 'x1,y1 x2,y2 x3,y3 x4,y4'.

        Returns:
            list: List of tuples containing the parsed coordinates.
        """
        coordinates = [tuple(map(int, coord.split(','))) for coord in coord_str.split()]
        return coordinates


    # Load your image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct perspective in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("coordinates", type=str, help="List of corner coordinates in the format 'x1,y1 x2,y2 x3,y3 x4,y4'.")
    args = parser.parse_args()

    
    image = cv2.imread(args.image_path)

    # Correct perspective
    coordinates = PerspectiveCorrection.parse_coordinates(args.coordinates)
    corrected_image = PerspectiveCorrection.correct_perspective(image, coordinates)

    # Display the corrected image
    cv2.imshow("Corrected Perspective", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
