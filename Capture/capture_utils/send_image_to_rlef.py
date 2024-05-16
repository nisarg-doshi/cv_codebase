import requests
import threading
import os
import argparse

class SendImageToRLEF:
    def __init__(self, model_id):
        """
        Initialize AutoAI client with model ID.
        
        Args:
            model_id (str): ID of the model.
        """
        self.model_id = model_id

    def send_image(self, model_name, label, file_path, tag, initial_confidence_score=100, prediction="initial", metadata="", delete_image_after_use=False):
        """
        Send an image to the AutoAI service for processing.
        
        Args:
            model_name (str): Name of the model.
            label (str): Label for the image.
            file_path (str): Path to the image file.
            tag (str): Tag for the image.
            initial_confidence_score (int): Initial confidence score (default: 100).
            prediction (str): Initial prediction (default: "initial").
            metadata (str): Additional metadata (default: "").
            delete_image_after_use (bool): Whether to delete the image after sending (default: False).
        """
        threading.Thread(target=self._send_image_to_autoai, args=(model_name, label, file_path, tag, initial_confidence_score, prediction, metadata, delete_image_after_use)).start()

    def _send_image_to_autoai(self, model_name, label, file_path, tag, initial_confidence_score, prediction, metadata, delete_image_after_use):
        """
        Helper function to send image to AutoAI service.
        
        Args:
            model_name (str): Name of the model.
            label (str): Label for the image.
            file_path (str): Path to the image file.
            tag (str): Tag for the image.
            initial_confidence_score (int): Initial confidence score.
            prediction (str): Initial prediction.
            metadata (str): Additional metadata.
            delete_image_after_use (bool): Whether to delete the image after sending.
        """
        try:
            url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"

            payload = {
                'status': 'backlog',
                'csv': metadata,
                'model': self.model_id,
                'label': label,
                'tag': tag,
                'confidence_score': initial_confidence_score,
                'prediction': prediction,
                'imageAnnotations': "",
                'model_type': 'image/png'
            }

            files = [('resource', (os.path.basename(file_path), open(file_path, 'rb'), 'image/png'))]
            headers = {}
            response = requests.post(url, headers=headers, data=payload, files=files, verify=False)
            if response.status_code == 200:
                print(f'Successfully sent to AutoAI: {model_name}....{file_path}')
            else:
                print('Error while sending to AutoAI')
        except Exception as e:
            print('Error while sending data to Auto AI:', e)
        finally:
            if delete_image_after_use:
                os.remove(file_path)
                print("Removed the image from local:", file_path)

if __name__ == "__main__":
    model_name = "Item In Hand Classification"
    model_id = "65b0f505ee58cd58dabc1b83"
    label = 'initial'
    file_path = "/home/onm/Downloads/image_for_ocr/image_camera_0_0.png"
    tag = 'top_shelf'
    client = SendImageToRLEF(model_id)
    client.send_image(model_name, label, file_path, tag)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send an image to RLEF for processing")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--model_id", type=str, help="ID of the model")
    parser.add_argument("--label", type=str, default="initial", help="Label for the image (default: 'initial')")
    parser.add_argument("--file_path", type=str, help="Path to the image file")
    parser.add_argument("--tag", type=str, default="top_shelf", help="Tag for the image (default: 'top_shelf')")
    args = parser.parse_args()

  
    client = SendImageToRLEF(args.model_id)
    client.send_image(args.model_name, args.label, args.file_path, args.tag)
