
# Import necessary libraries
from ultralytics import YOLO

def test_yolov8_model(model_path, image_path):
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

if __name__ == "__main__":

    # Testing parameters
    model_path = "best.pt"
    image_path = "image.png"

    # Test YOLOv8 model
    test_yolov8_model(model_path, image_path)
