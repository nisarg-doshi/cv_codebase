import os
import cv2
import torch
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import supervision as sv
import numpy as np 
from flask import Flask, request, jsonify
import time
import datetime
import random
from google.cloud import storage

from google.cloud import storage
import argparse

class GCSUploader:
    """
    A class to upload files to Google Cloud Storage (GCS).
    """

    def __init__(self, key_path):
        """
        Initializes the GCSUploader object.

        Args:
            key_path (str): The path to the service account JSON key file.
        """
        self.key_path = key_path
        self.storage_client = storage.Client.from_service_account_json(self.key_path)

    def upload_file(self, gcs_path, local_path, bucket_name):
        """
        Uploads a file to Google Cloud Storage (GCS).

        Args:
            gcs_path (str): The path to the destination file in GCS.
            local_path (str): The path to the local file to upload.
            bucket_name (str): The name of the GCS bucket to upload to.

        Returns:
            str: A message indicating whether the upload was successful or not.
        """
        # Get the bucket object
        bucket = self.storage_client.bucket(bucket_name)

        # Create a blob object representing the destination file in GCS
        blob = bucket.blob(gcs_path)

        # Upload the file to GCS
        blob.upload_from_filename(local_path)

        # Return a success message
        return "Upload Successful"


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Google Cloud Storage File Uploader")
#     parser.add_argument("--key_path", required=True, help="Path to the service account JSON key file")
#     parser.add_argument("--gcs_path", required=True, help="Path to the destination file in GCS")
#     parser.add_argument("--local_path", required=True, help="Path to the local file to upload")
#     parser.add_argument("--bucket_name", required=True, help="Name of the GCS bucket to upload to")
#     args = parser.parse_args()

#     uploader = GCSUploader(args.key_path)
#     result = uploader.upload_file(args.gcs_path, args.local_path, args.bucket_name)
#     print(result)





# Function to generate a signed URL for a file in Google Cloud Storage
class GCSSignedURLGenerator:
    """
    A class to generate signed URLs for downloading files from Google Cloud Storage (GCS).
    """

    def __init__(self, key_path):
        """
        Initializes the GCSSignedURLGenerator object.

        Args:
            key_path (str): The path to the service account JSON key file.
        """
        self.key_path = key_path
        self.storage_client = storage.Client.from_service_account_json(self.key_path)

    def generate_signed_url(self, name, bucket_name, expiration_hours=10):
        """
        Generates a signed URL for downloading a file from Google Cloud Storage (GCS).

        Args:
            name (str): The name of the file in GCS.
            bucket_name (str): The name of the GCS bucket containing the file.
            expiration_hours (int): Expiration time for the signed URL in hours (default: 10).

        Returns:
            str: The signed URL for downloading the file.
        """
        # Get the bucket object
        bucket = self.storage_client.bucket(bucket_name)

        # Get the blob object representing the file in GCS
        blob = bucket.blob(name)

        # Generate a signed URL with the specified expiration time
        expiration_time = datetime.timedelta(hours=expiration_hours)
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET"
        )

        # Return the signed URL
        return url
# Function to enhance class names
# Function to enhance class names by adding "all" prefix
def enhance_class_name(class_names: List[str]) -> List[str]:
    """
    Enhance class names by adding "all" prefix.

    Args:
    - class_names (List[str]): List of class names.

    Returns:
    - List[str]: List of enhanced class names.
    """
    return [f"all {class_name}s" for class_name in class_names]

# Function to segment objects in an image
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Segment objects in an image using SAM model.

    Args:
    - sam_predictor (SamPredictor): SAM model predictor.
    - image (np.ndarray): Input image.
    - xyxy (np.ndarray): Bounding boxes of detected objects.

    Returns:
    - np.ndarray: Segmented masks for detected objects.
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# Function to perform inference on an image
def inference_model(img_id, image, class_list=["person"]):
    """
    Perform inference on an image.

    Args:
    - img_id (str): Image ID.
    - image (np.ndarray): Input image.
    - class_list (List[str]): List of classes for object detection.

    Returns:
    - Tuple[np.ndarray, str]: Annotated image and path to the saved image.
    """
    # Paths to model configurations and checkpoints
    GROUNDING_DINO_CONFIG_PATH = os.path.join("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
    SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")

    # Initialize grounding DINO model and SAM model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    sam_predictor = SamPredictor(sam)

    # Perform object detection
    CLASSES = class_list
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=0.35,
        text_threshold=0.25
    )

    # Convert detections to masks
    detections.mask = segment(sam_predictor, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)

    # Create annotated image
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [f"{CLASSES[0]} {confidence:0.2f}" for _,_,confidence, class_id, _ in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Save annotated image
    filename = f"{img_id}_annoted.png"
    path = os.path.join('UPLOAD_FOLDER', filename)
    cv2.imwrite(path, annotated_image)
    
    return annotated_image, path




# Main function to run the Flask application
if __name__ == '__main__':
    image_path = "temp.png"
    image_id = random.randint(11,999999)
    class_list = ["person"]
    image = cv2.imread(image_path)
    
    annotated_image, path = inference_model(image_id, image, class_list)
            
    
    
    print("background subtarcted image saved at {path}")
