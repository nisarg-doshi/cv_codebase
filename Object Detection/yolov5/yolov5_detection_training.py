import os
import cv2
import json
import yaml
from ultralytics import YOLO
import argparse


class JSONConverter:
    """
    Converts JSON annotations to YOLO format text files and creates a YAML file for YOLO training.
    """

    def __init__(self, json_file, dataset_dir, training_id=123):
        self.json_file = json_file
        self.dataset_dir = dataset_dir
        self.training_id = training_id

    def convert_to_yolo_format(self):
        """
        Converts JSON annotations to YOLO format.

        Returns:
        None
        """
        print("Json conversion started")

        # Load JSON annotations
        jsfile = json.load(open(self.json_file, "r"))

        # Dictionary to map image IDs to file names
        image_id = {}
        for image in jsfile["images"]:
            image_id[image['id']] = image['file_name']

        # Iterate through annotations
        for itr in range(len(jsfile["annotations"])):
            ann = jsfile["annotations"][itr]
            poly = ann["segmentation"][0]
            img = cv2.imread(self.dataset_dir + "/images/" + image_id[ann["image_id"]])

            # Skip if image cannot be read
            try:
                height, width, depth = img.shape
            except:
                continue

            xmin = 999
            ymin = 999
            xmax = -1
            ymax = -1

            for i in range(len(poly) // 2):
                xmin = min(xmin, poly[2 * i])
                xmax = max(xmax, poly[2 * i])
                ymin = min(ymin, poly[2 * i + 1])
                ymax = max(ymax, poly[2 * i + 1])

            bbox = [ann["category_id"], (xmax + xmin) / (2 * width), (ymax + ymin) / (2 * height),
                    (xmax - xmin) / width,
                    (ymax - ymin) / height]

            label_dir = os.path.join(self.dataset_dir, "labels")
            os.makedirs(label_dir, exist_ok=True)

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

        classes = {i["id"]: i["name"] for i in jsfile["categories"]}

        yaml_file = {
            "train": f"{str(os.getcwd())}/datasets/{self.training_id}/images",
            "val": f"{str(os.getcwd())}/datasets/test_{self.training_id}/images"
        }

        yaml_file["nc"] = len(classes)
        yaml_file["names"] = classes

        yaml_file_path = os.path.join("datasets", f"{self.training_id}.yaml")
        file = open(yaml_file_path, "w")
        yaml.dump(yaml_file, file)


class YOLOTrainer:
    """
    Trains a YOLOv5 model using the specified YAML file, number of epochs, batch size, and device.
    """

    def __init__(self, yaml_path, epochs, batch, device):
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.batch = batch
        self.device = device

    def train_yolo_model(self):
        """
        Trains a YOLOv5 model.

        Returns:
        None
        """
        # Initialize YOLOv5 model
        model = YOLO('yolov5n.pt')  # Load a pretrained model (recommended for training)

        # Train the model
        model.train(
            data=self.yaml_path,
            epochs=self.epochs,
            batch=self.batch,
            device=self.device
        )





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a YOLOv5 model.")
    parser.add_argument("--json_file", type=str, help="Path to the JSON file containing annotations.")
    parser.add_argument("--dataset_dir", type=str, help="Directory containing the dataset images.")
    parser.add_argument("--training_id", type=int, default=123, help="Identifier for the training dataset.")
    parser.add_argument("--yaml_path", type=str, default="datasets/123.yaml",
                        help="Path to the YAML file containing dataset information.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use for training (e.g., 'cpu', 'gpu').")
    args = parser.parse_args()

    # Convert JSON annotations to YOLO format
    json_converter = JSONConverter(args.json_file, args.dataset_dir, args.training_id)
    json_converter.convert_to_yolo_format()

    # Train YOLOv5 model
    yolo_trainer = YOLOTrainer(args.yaml_path, args.epochs, args.batch, args.device)
    yolo_trainer.train_yolo_model()
