import requests
import threading
import os
import shutil
import argparse
class SendVideoToRLEF:
    def __init__(self, model_id):
        """
        Initialize AutoAI client with model ID.
        
        Args:
            model_id (str): ID of the model.
        """
        self.model_id = model_id
        self.final_path = self.create_output_directories()

    def __del__(self):
        shutil.rmtree(self.final_path)
        pass
    
    def create_output_directories(self):
        final_video_directory = 'converted_videos'
        if not os.path.exists(final_video_directory):
            os.makedirs(final_video_directory)

        return final_video_directory
        
    def convert_video(self, input_path, output_path):
        os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

    def send_video(self, model_name, label, file_path, tag, initial_confidence_score=100, prediction="initial", metadata="", delete_after_use=False):
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
        self.file_path = file_path
        self.final_file_path = os.path.join(self.final_path, file_path.split('/')[-1]) 
        self.convert_video(file_path, self.final_file_path)
        threading.Thread(target=self._send_video_to_autoai, args=(model_name, label, self.final_file_path, tag, initial_confidence_score, prediction, metadata, delete_after_use)).start()

    def _send_video_to_autoai(self, model_name, label, file_path, tag, initial_confidence_score, prediction, metadata, delete_after_use):
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
                'model': self.model_id,
                'status': 'backlog',
                'csv': metadata,
                'label': label,
                'tag': tag,
                'model_type': 'video',
                'prediction': prediction,
                'confidence_score': initial_confidence_score,
                'appShouldNotUploadResourceFileToGCS': 'true',
                'resourceFileName': file_path,
                'resourceContentType': "video/mp4"
            }

            headers = {}

            response = requests.request("POST", url, headers=headers, data=payload)

            headers = {'Content-Type': 'video/mp4'}

            print(response.status_code)

            api_url_upload = response.json()["resourceFileSignedUrlForUpload"]

            response = requests.request("PUT", api_url_upload, headers=headers, data=open(os.path.join(file_path), 'rb'))
            

            if response.status_code == 200:
                print(f'Successfully sent to AutoAI: {model_name}....{file_path}')
            else:
                print('Error while sending to AutoAI')

        except Exception as e:
            print('Error while sending data to Auto AI:', e)
        finally:
            if delete_after_use:   
                os.remove(self.file_path)
                print("Removed the converted video:", self.file_path)

            # PS: it does not delete the original source file, it only deletes the newly converted video locally which is sent to rlef.
            # self.file_path is the original video path and file_path local variable is path for converted video
            # os.remove(file_path)
            # print("Removed the converted video:", file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a video to RLEF for processing")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--model_id", type=str, help="ID of the model")
    parser.add_argument("--label", type=str, default="initial", help="Label for the video (default: 'initial')")
    parser.add_argument("--file_path", type=str, help="Path to the video file")
    parser.add_argument("--tag", type=str, default="top_shelf", help="Tag for the video (default: 'top_shelf')")
    args = parser.parse_args()



    client = SendVideoToRLEF(args.model_id)
    client.send_video(args.model_name, args.label, args.file_path, args.tag)
    del client  # To cleanup dir created

