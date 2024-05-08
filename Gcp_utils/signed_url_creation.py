from google.cloud import storage
import datetime

def create_signedurl(name, bucket, key_path):
    """
    Generates a signed URL for downloading a file from Google Cloud Storage (GCS).

    Args:
        name (str): The name of the file in GCS.
        bucket (str): The name of the GCS bucket containing the file.
        key_path (str): The path to the service account JSON key file.

    Returns:
        str: The signed URL for downloading the file.
    """
    # Initialize the storage client using the service account key
    storage_client = storage.Client.from_service_account_json(key_path)
    
    # Get the bucket object
    bucket = storage_client.bucket(bucket)
    
    # Get the blob object representing the file in GCS
    blob = bucket.blob(name)
    
    # Generate a signed URL with an expiration time of 10 hours
    url = blob.generate_signed_url(
        version="v4", 
        expiration=datetime.timedelta(hours=10), 
        method="GET"
    )
    
    # Return the signed URL
    return url
