import cv2
import numpy as np
import argparse

class ImageProcessing:
    """
    Class for various image processing operations.
    """

    @staticmethod
    def normalize(img):
        """
        Normalize pixel values to the [0, 1] range.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Normalized image.
        """
        return img.astype(np.float32) / 255.0

    @staticmethod
    def remove_artifacts(img, kernel_size=(3, 3), iterations=1):
        """
        Remove salt-and-pepper noise (artifacts) using morphological opening.

        Args:
            img (numpy.ndarray): Input image.
            kernel_size (tuple): Kernel size for morphological operation.
            iterations (int): Number of iterations for morphological operation.

        Returns:
            numpy.ndarray: Image with artifacts removed.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

    @staticmethod
    def gaussian_filter(img):
        """
        Apply Gaussian filtering.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image after Gaussian filtering.
        """
        return cv2.GaussianBlur(img, (5, 5), 0)

    @staticmethod
    def median_filter(img):
        """
        Apply Median filtering.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image after Median filtering.
        """
        return cv2.medianBlur(img, 5)

    @staticmethod
    def histogram_equalization(img):
        """
        Apply histogram equalization.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image after histogram equalization.
        """
        return cv2.equalizeHist(img)

    @staticmethod
    def thresholding(img, thresh=127, max_val=255, thresh_type=cv2.THRESH_BINARY):
        """
        Apply thresholding.

        Args:
            img (numpy.ndarray): Input image.
            thresh (float): Threshold value.
            max_val (float): Maximum value to use with the thresholding type.
            thresh_type (int): Type of thresholding (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, etc.).

        Returns:
            numpy.ndarray: Thresholded image.
        """
        _, thresh_img = cv2.threshold(img, thresh, max_val, thresh_type)
        return thresh_img

    @staticmethod
    def erosion(img, kernel_size=(3, 3), iterations=1):
        """
        Apply erosion morphological operation.

        Args:
            img (numpy.ndarray): Input image.
            kernel_size (tuple): Kernel size for the structuring element.
            iterations (int): Number of iterations for erosion.

        Returns:
            numpy.ndarray: Image after erosion operation.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.erode(img, kernel, iterations=iterations)

    @staticmethod
    def dilation(img, kernel_size=(3, 3), iterations=1):
        """
        Apply dilation morphological operation.

        Args:
            img (numpy.ndarray): Input image.
            kernel_size (tuple): Kernel size for the structuring element.
            iterations (int): Number of iterations for dilation.

        Returns:
            numpy.ndarray: Image after dilation operation.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        return cv2.dilate(img, kernel, iterations=iterations)

def main(input_img_path):
    """
    Main function for image processing operations.

    Args:
        input_img_path (str): Path to the input image.
    """
    # Read input image
    img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    # Perform image processing operations
    processed_img = ImageProcessing.normalize(img)
    processed_img = ImageProcessing.remove_artifacts(processed_img)
    processed_img = ImageProcessing.gaussian_filter(processed_img)
    processed_img = ImageProcessing.median_filter(processed_img)
    processed_img = ImageProcessing.histogram_equalization(processed_img)
    processed_img = ImageProcessing.thresholding(processed_img)
    processed_img = ImageProcessing.erosion(processed_img)
    processed_img = ImageProcessing.dilation(processed_img)

    # Display processed image
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform various image processing operations.")
    parser.add_argument("input_img_path", type=str, help="Path to the input image.")
    args = parser.parse_args()
    main(args.input_img_path)
