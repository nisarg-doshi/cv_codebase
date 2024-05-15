
# Import necessary libraries
import argparse

from ultralytics import YOLO

def inference(model_path, image_path):
    """
    Tests a YOLOv8 model on a single image and displays the results.

    Args:
    - model_path (str): Path to the YOLOv8 model file.
    - image_path (str): Path to the image to be tested.

    Returns:
    None
    """
    # Initialize YOLOv8 model
    model = YOLO(model_path)

    # Perform inference on the image
    results = model(image_path)

    # Extract and display the results
    for result in results:
        result.show()
        result.save(filename='result.jpg')


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv5 model on a single image")
    parser.add_argument("--model_path", type=str, required=True, default = "yolov8n.pt", help="Path to the YOLOv5 model file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be tested")
    args = parser.parse_args()

    # Test YOLOv8 model
    inference(args.model_path, args.image_path)

if __name__ == "__main__":
    main()
