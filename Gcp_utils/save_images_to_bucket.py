from google.cloud import storage
import argparse

class GCSUploader:
    """
    A class to upload files to Google Cloud Storage (GCS).
    """

    def __init__(self, key_path):
        """
        Initializes the GCSUploader object.

        Args:
            key_path (str): The path to the service account JSON key file.
        """
        self.key_path = key_path
        self.storage_client = storage.Client.from_service_account_json(self.key_path)

    def upload_file(self, gcs_path, local_path, bucket_name):
        """
        Uploads a file to Google Cloud Storage (GCS).

        Args:
            gcs_path (str): The path to the destination file in GCS.
            local_path (str): The path to the local file to upload.
            bucket_name (str): The name of the GCS bucket to upload to.

        Returns:
            str: A message indicating whether the upload was successful or not.
        """
        # Get the bucket object
        bucket = self.storage_client.bucket(bucket_name)

        # Create a blob object representing the destination file in GCS
        blob = bucket.blob(gcs_path)

        # Upload the file to GCS
        blob.upload_from_filename(local_path)

        # Return a success message
        return "Upload Successful"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Cloud Storage File Uploader")
    parser.add_argument("--key_path", required=True, help="Path to the service account JSON key file")
    parser.add_argument("--gcs_path", required=True, help="Path to the destination file in GCS")
    parser.add_argument("--local_path", required=True, help="Path to the local file to upload")
    parser.add_argument("--bucket_name", required=True, help="Name of the GCS bucket to upload to")
    args = parser.parse_args()

    uploader = GCSUploader(args.key_path)
    result = uploader.upload_file(args.gcs_path, args.local_path, args.bucket_name)
    print(result)



