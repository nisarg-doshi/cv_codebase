import os
import cv2
import json
import yaml
from ultralytics import YOLO
import argparse

class JSONToYOLOConverter:
    """
    Class for converting JSON annotations to YOLO format and creating YAML file for YOLO training.
    """

    def __init__(self, training_id=123):
        self.training_id = training_id

    def convert_json_to_yolo(self, json_file, dataset_dir):
        """
        Converts JSON annotations to YOLO format text files and creates a YAML file for YOLO training.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            dataset_dir (str): Directory containing the dataset images.

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
            "train": f"{str(os.getcwd())}/datasets/{self.training_id}/images",
            "val": f"{str(os.getcwd())}/datasets/test_{self.training_id}/images"
        }
        yaml_file["nc"] = len(classes)
        yaml_file["names"] = classes

        # Write YAML file
        yaml_file_path = os.path.join("datasets", f"{self.training_id}.yaml")
        file = open(yaml_file_path, "w")
        yaml.dump(yaml_file, file)

class YOLOTrainer:
    """
    Class for training a YOLOv8 model using the specified YAML file, number of epochs, batch size, and device.
    """

    def train_yolo_model(self, yaml_path, epochs, batch, device):
        """
        Trains a YOLOv8 model using the specified YAML file, number of epochs, batch size, and device.

        Args:
            yaml_path (str): Path to the YAML file containing dataset information.
            epochs (int): Number of epochs for training.
            batch (int): Batch size for training.
            device (str): Device to use for training (e.g., 'cpu', 'gpu').

        Returns:
            None
        """
        # Initialize YOLOv8 model
        model = YOLO('yolov8x-seg.pt')  # Load a pretrained model (recommended for training)

        # Train the model
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            device=device
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model using COCO JSON annotations.")
    parser.add_argument("json_file", type=str, help="Path to the COCO JSON file containing annotations.")
    parser.add_argument("dataset_dir", type=str, help="Directory containing the dataset images.")
    parser.add_argument("--training_id", type=int, default=123, help="Identifier for the training dataset (default is 123).")
    parser.add_argument("--yaml_path", type=str, default="datasets/123.yaml", help="Path to the YAML file containing dataset information.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use for training (e.g., 'cpu', 'gpu').")
    args = parser.parse_args()

    # Convert JSON annotations to YOLO format
    converter = JSONToYOLOConverter(training_id=args.training_id)
    converter.convert_json_to_yolo(json_file=args.json_file, dataset_dir=args.dataset_dir)

    # Train YOLOv8 model
    trainer = YOLOTrainer()
    trainer.train_yolo_model(yaml_path=args.yaml_path, epochs=args.epochs, batch=args.batch, device=args.device)

