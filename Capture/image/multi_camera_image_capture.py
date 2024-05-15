import cv2
import numpy as np
import os
import uuid
import argparse
from Capture.capture_utils.send_image_to_rlef import SendImageToRLEF  # Assuming SendImageToRLEF class exists

class MultiCameraImageCapture:
    """
    A class to capture images from multiple cameras simultaneously.

    Attributes:
    camera_indexes (list): List of camera indexes for each camera device.
    image_width (int): The width of the captured image (default is 640).
    image_height (int): The height of the captured image (default is 480).
    destination_folder (str): The folder path to save captured images (default is current directory).
    rlef_model_id (str): The model ID for RLEF integration.
    """

    def __init__(self, camera_indexes, image_width=640, image_height=480, destination_folder="", rlef_model_id=""):
        """
        Initializes the MultiCameraImageCapture object with the given camera indexes, width, height, destination folder, and RLEF model ID.

        Parameters:
        camera_indexes (list): List of camera indexes for each camera device.
        image_width (int): The width of the captured image (default is 640).
        image_height (int): The height of the captured image (default is 480).
        destination_folder (str): The folder path to save captured images (default is current directory).
        rlef_model_id (str): The model ID for RLEF integration.
        """
        self.camera_indexes = camera_indexes
        self.image_width = image_width
        self.image_height = image_height
        self.destination_folder = destination_folder
        self.rlef_model_id = rlef_model_id

    def capture_frames_and_send_to_rlef(self, model_name="", label="", tags=[], initial_confidence_score=100, prediction="initial", metadata="", delete_image_after_use=False):
        """
        Captures frames from multiple cameras simultaneously and sends them to RLEF for processing.

        Parameters:
        model_name (str): The name of the RLEF model.
        label (str): The label for the frames.
        tags (list): List of tags for the frames corresponding to each camera.
        initial_confidence_score (int): The initial confidence score for the frames.
        prediction (str): The prediction for the frames.
        metadata (str): Additional metadata for the frames.
        delete_image_after_use (bool): Whether to delete the frames after sending them to RLEF.
        """
        if len(tags) != len(self.camera_indexes):
            raise ValueError("Number of tags must match the number of cameras.")

        caps = [cv2.VideoCapture(index) for index in self.camera_indexes]
        for cap in caps:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        if self.rlef_model_id:
            client = SendImageToRLEF(self.rlef_model_id)

        counter = 0
        while True:
            frames = [cap.read()[1] for cap in caps]
            concatenated_frame = np.hstack(frames)
            cv2.imshow('Press "c" to capture, press "q" to quit', cv2.resize(concatenated_frame, (self.image_width * len(self.camera_indexes), self.image_height)))
            key = cv2.waitKey(1)

            if key == ord('c'):
                for i, frame in enumerate(frames):
                    random_name = str(uuid.uuid4())
                    image_path = os.path.join(self.destination_folder, f"camera_{i}_frame_{counter}_{random_name}.png")
                    cv2.imwrite(image_path, frame)
                    print(f"Frame captured: {image_path}")
                    if self.rlef_model_id:
                        client.send_image(model_name, label, image_path, tags[i], initial_confidence_score, prediction, metadata, delete_image_after_use)
                counter += 1

            elif key == ord('q'):
                break

        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    def capture_frames_locally(self):
        """
        Captures frames from multiple cameras simultaneously and saves them locally without sending them to RLEF.
        """
        caps = [cv2.VideoCapture(index) for index in self.camera_indexes]
        for cap in caps:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)

        if self.destination_folder:
            os.makedirs(self.destination_folder, exist_ok=True)

        camera_folders = [os.path.join(self.destination_folder, f"camera_{i}") for i in range(len(self.camera_indexes))]
        for folder in camera_folders:
            os.makedirs(folder, exist_ok=True)

        counter = 0
        while True:
            frames = [cap.read()[1] for cap in caps]
            concatenated_frame = np.hstack(frames)
            cv2.imshow('Press "c" to capture, press "q" to quit', cv2.resize(concatenated_frame, (self.image_width * len(self.camera_indexes), self.image_height)))
            key = cv2.waitKey(1)

            if key == ord('c'):
                for i, frame in enumerate(frames):
                    random_name = str(uuid.uuid4())
                    image_path = os.path.join(camera_folders[i], f"camera_{i}_frame_{counter}_{random_name}.png")
                    cv2.imwrite(image_path, frame)
                    print(f"Frame captured locally: {image_path}")
                counter += 1

            elif key == ord('q'):
                break

        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-camera image capture with optional sending to RLEF")
    parser.add_argument("--rlef_model_id", type=str, default="", help="ID of the RLEF model (default: '')")
    parser.add_argument("--camera_indexes", nargs="+", type=int, default=[2, 5, 4], help="Indexes of the cameras (default: [2, 5, 4])")
    parser.add_argument("--image_width", type=int, default=640, help="Width of the captured images (default: 640)")
    parser.add_argument("--image_height", type=int, default=480, help="Height of the captured images (default: 480)")
    parser.add_argument("--destination_folder", type=str, default="", help="Destination folder for saving images locally (default: '')")
    args = parser.parse_args()

    rlef_model_id = args.rlef_model_id
    camera_indexes = args.camera_indexes
    image_width = args.image_width
    image_height = args.image_height
    destination_folder = args.destination_folder

    multi_camera_capture = MultiCameraImageCapture(
        camera_indexes=camera_indexes,
        image_width=image_width,
        image_height=image_height,
        destination_folder=destination_folder,
        rlef_model_id=rlef_model_id
    )

    # Example usage for capturing frames and sending to RLEF
    if rlef_model_id:
        model_name = "Item In Hand Classification"
        label = 'initial'
        tags = ['top_left_shelf', 'top_right_shelf', 'bottom_shelf']
        multi_camera_capture.capture_frames_and_send_to_rlef(model_name, label, tags)

    # Example usage for capturing frames locally
    else:
        multi_camera_capture.capture_frames_locally()

