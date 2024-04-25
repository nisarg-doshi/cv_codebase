import cv2
import os
import random
from Capture.rlef_utils.send_image_to_rlef import SendImageToRLEF

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
    rlef_model_id = "65b0f505ee58cd58dabc1b83"
    single_image_capture = SingleImageCapture(camera_index=5, image_width=1280, image_height=960, destination_folder="", rlef_model_id=rlef_model_id)
    
    if rlef_model_id=="":  # Only saves the images locally
        single_image_capture.capture_images()
    else:  # Sends it to rlef if model id mentioned
        model_name = "Item In Hand Classification"
        label = 'initial'
        tag = 'top_shelf'
        single_image_capture.capture_images(model_name, label, tag)
