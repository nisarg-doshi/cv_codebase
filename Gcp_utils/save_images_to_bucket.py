from google.cloud import storage

def upload_file_to_bucket(gcs_path, local_path, bucket, key_path):
    """
    Uploads a file to Google Cloud Storage (GCS).

    Args:
        gcs_path (str): The path to the destination file in GCS.
        local_path (str): The path to the local file to upload.
        bucket (str): The name of the GCS bucket to upload to.
        key_path(str): The path to the service account JSON key file.

    Returns:
        str: A message indicating whether the upload was successful or not.
    """
    # Initialize the storage client using the service account key
    storage_client = storage.Client.from_service_account_json(key)
    
    # Get the bucket object
    bucket = storage_client.bucket(bucket)
    
    # Create a blob object representing the destination file in GCS
    blob = bucket.blob(gcs_path)
    
    # Upload the file to GCS
    blob.upload_from_filename(local_path)
    
    # Return a success message
    return "Upload Successful"
