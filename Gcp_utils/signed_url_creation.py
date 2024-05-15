from google.cloud import storage
import datetime
import argparse

class GCSSignedURLGenerator:
    """
    A class to generate signed URLs for downloading files from Google Cloud Storage (GCS).
    """

    def __init__(self, key_path):
        """
        Initializes the GCSSignedURLGenerator object.

        Args:
            key_path (str): The path to the service account JSON key file.
        """
        self.key_path = key_path
        self.storage_client = storage.Client.from_service_account_json(self.key_path)

    def generate_signed_url(self, name, bucket_name, expiration_hours=10):
        """
        Generates a signed URL for downloading a file from Google Cloud Storage (GCS).

        Args:
            name (str): The name of the file in GCS.
            bucket_name (str): The name of the GCS bucket containing the file.
            expiration_hours (int): Expiration time for the signed URL in hours (default: 10).

        Returns:
            str: The signed URL for downloading the file.
        """
        # Get the bucket object
        bucket = self.storage_client.bucket(bucket_name)

        # Get the blob object representing the file in GCS
        blob = bucket.blob(name)

        # Generate a signed URL with the specified expiration time
        expiration_time = datetime.timedelta(hours=expiration_hours)
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET"
        )

        # Return the signed URL
        return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Cloud Storage Signed URL Generator")
    parser.add_argument("--key_path", required=True, help="Path to the service account JSON key file")
    parser.add_argument("--name", required=True, help="Name of the file in GCS")
    parser.add_argument("--bucket_name", required=True, help="Name of the GCS bucket containing the file")
    parser.add_argument("--expiration_hours", type=int, default=10, help="Expiration time for the signed URL in hours (default: 10)")
    args = parser.parse_args()

    url_generator = GCSSignedURLGenerator(args.key_path)
    signed_url = url_generator.generate_signed_url(args.name, args.bucket_name, args.expiration_hours)
    print("Signed URL:", signed_url)



