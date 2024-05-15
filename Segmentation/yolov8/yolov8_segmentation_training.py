# Import necessary libraries
import os
import cv2
import json
import yaml
from ultralytics import YOLO

# Function to convert JSON annotations to YOLO format text files and create a YAML file for YOLO training
def json_to_txt_convert(json_file, dataset_dir, training_id=123):
    """
    Converts JSON annotations to YOLO format text files and creates a YAML file for YOLO training.

    Args:
    - json_file (str): Path to the JSON file containing annotations.
    - dataset_dir (str): Directory containing the dataset images.
    - training_id (int): Identifier for the training dataset (default is 123).

    Returns:
    None
    """


    print("Json conversion started")
    
    # Load JSON annotations
    jsfile = json.load(open(json_file, "r"))

    # Dictionary to map image IDs to file names
    image_id = {}
    for image in jsfile["images"]:
        image_id[image['id']] = image['file_name']

    # Iterate through annotations
    for itr in range(len(jsfile["annotations"])):
        ann = jsfile["annotations"][itr]
        poly = ann["segmentation"][0]
        img = cv2.imread(dataset_dir + "/images/" + image_id[ann["image_id"]])
        
        # Skip if image cannot be read
        try:
            height, width, depth = img.shape
        except:
            continue

        # Convert annotations to YOLO format
        bbox = [ann["category_id"]]
        for i in range(len(poly) // 2):
            _ = poly[2 * i] / width
            bbox.append(_)
            _ = poly[2 * i + 1] / height
            bbox.append(_)

        # Create label directory if it doesn't exist
        label_dir = os.path.join(dataset_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)

        # Write annotations to text files
        if os.path.exists(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt")):
            file = open(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt"), "a")
            file.write("\n")
            file.write(" ".join(map(str, bbox)))
        else:
            file = open(os.path.join(
                label_dir, os.path.splitext(os.path.basename(image_id[ann["image_id"]]))[0] + ".txt"), "a")
            file.write(" ".join(map(str, bbox)))
        file.close()

    # Extract classes from JSON and create YAML file for YOLO training
    classes = {i["id"]: i["name"] for i in jsfile["categories"]}
    yaml_file = {
        "train": f"{str(os.getcwd())}/datasets/{training_id}/images",
        "val": f"{str(os.getcwd())}/datasets/test_{training_id}/images"
    }
    yaml_file["nc"] = len(classes)
    yaml_file["names"] = classes

    # Write YAML file
    yaml_file_path = os.path.join("datasets", f"{training_id}.yaml")
    file = open(yaml_file_path, "w")
    yaml.dump(yaml_file, file)

def train_yolo8_model(yaml_path, epochs, batch, device, json_file):
    """
    Trains a YOLOv8 model using the specified YAML file, number of epochs, batch size, and device.

    Args:
    - yaml_path (str): Path to the YAML file containing dataset information.
    - epochs (int): Number of epochs for training.
    - batch (int): Batch size for training.
    - device (str): Device to use for training (e.g., 'cpu', 'gpu').
    - json_file (str): Path to the COCO JSON file containing annotations.

    Returns:
    None
    """
    # Convert COCO JSON annotations to YOLO format
    json_to_txt_convert(dataset_dir="datasets", json_file=json_file)

    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch,
        device=device
    )

if __name__ == "__main__":
    # Training parameters
    yaml_path = "datasets/123.yaml"
    epochs = 100
    batch = 32
    device = "gpu"
    json_file = "coco.json"

    # Train YOLOv8 model
    train_yolo8_model(yaml_path, epochs, batch, device, json_file)

