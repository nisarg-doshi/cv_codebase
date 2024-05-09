import cv2
import numpy as np

def normalize(img):
    """Normalize pixel values to [0, 1] range"""
    return img.astype(np.float32) / 255.0

def remove_artifacts(img, kernel_size=(3, 3), iterations=1):
    """Remove salt-and-pepper noise (artifacts)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

def gaussian_filter(img):
    """Apply Gaussian filtering"""
    return cv2.GaussianBlur(img, (5, 5), 0)

def median_filter(img):
    """Apply Median filtering"""
    return cv2.medianBlur(img, 5)

def histogram_equalization(img):
    """Apply histogram equalization"""
    return cv2.equalizeHist(img)

def thresholding(img, thresh=127, max_val=255, thresh_type=cv2.THRESH_BINARY):
    """Apply thresholding"""
    _, thresh_img = cv2.threshold(img, thresh, max_val, thresh_type)
    return thresh_img

def erosion(img, kernel_size=(3, 3), iterations=1):
    """Apply erosion morphological operation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.erode(img, kernel, iterations=iterations)

def dilation(img, kernel_size=(3, 3), iterations=1):
    """Apply dilation morphological operation"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)