import cv2
import os
import random
from Capture.capture_utils.send_image_to_rlef import SendImageToRLEF
import argparse
class SingleImageCapture:
    """
    A class to capture images from a camera.
    
    Attributes:
    camera_index (int): The index of the camera device.
    image_width (int): The width of the captured image (default is 640).
    image_height (int): The height of the captured image (default is 480).
    """

    def __init__(self, camera_index, image_width=640, image_height=480, destination_folder="", rlef_model_id=""):
        """
        Initializes the SingleImageCapture object with the given camera index, width, and height.
        
        Parameters:
        camera_index (int): The index of the camera device.
        image_width (int): The width of the captured image (default is 640).
        image_height (int): The height of the captured image (default is 480).
        """
        self.camera_index = camera_index
        self.image_width = image_width
        self.image_height = image_height
        self.destination_folder = destination_folder
        self.rlef_model_id = rlef_model_id

    def capture_images(self, model_name="", label="", tag="", initial_confidence_score=100, prediction="initial", metadata="", delete_image_after_use=False):
        """
        Captures images from the camera and saves them to the specified destination folder.
        
        Parameters:
        destination_folder (str): The folder path to save captured images (default is current directory).
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        if self.destination_folder:
            os.makedirs(self.destination_folder, exist_ok=True)
        
        if self.rlef_model_id:
            client = SendImageToRLEF(self.rlef_model_id)

        while True:
            ret, frame = cap.read()
            cv2.imshow('Press "c" to capture, press "q" to quit', cv2.resize(frame, (self.image_width, self.image_height)))
            key = cv2.waitKey(1)
            
            if key == ord('c'):
                image_path = os.path.join(self.destination_folder, f"image_{random.randint(1000000, 99999999)}.png")
                cv2.imwrite(image_path, frame)
                print(f"Image captured: {image_path}")
                if self.rlef_model_id:
                    client.send_image(model_name, label, image_path, tag, initial_confidence_score, prediction, metadata, delete_image_after_use)
                
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Example Usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single camera image capture with optional sending to RLEF")
    parser.add_argument("--rlef_model_id", type=str, default="", help="ID of the RLEF model (default: '')")
    parser.add_argument("--camera_index", type=int, default=5, help="Index of the camera (default: 5)")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of the captured image (default: 1280)")
    parser.add_argument("--image_height", type=int, default=960, help="Height of the captured image (default: 960)")
    parser.add_argument("--destination_folder", type=str, default="", help="Destination folder for saving images locally (default: '')")
    args = parser.parse_args()

    rlef_model_id = args.rlef_model_id
    camera_index = args.camera_index
    image_width = args.image_width
    image_height = args.image_height
    destination_folder = args.destination_folder

    single_image_capture = SingleImageCapture(
        camera_index=camera_index,
        image_width=image_width,
        image_height=image_height,
        destination_folder=destination_folder,
        rlef_model_id=rlef_model_id
    )

    # If no RLEF model ID is provided, only save images locally
    if rlef_model_id == "":
        single_image_capture.capture_images()
    else:
        model_name = "Item In Hand Classification"
        label = 'initial'
        tag = 'top_shelf'
        single_image_capture.capture_images(model_name, label, tag)

