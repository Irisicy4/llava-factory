import boto3
from PIL import Image
from io import BytesIO
import os
import traceback
import json

def test_s3_connection(data_args, image_files):
    """
    Test loading images from S3 (or local if not s3://).
    
    Args:
        data_args: object with attribute `image_folder`
        image_files: list of image file paths/keys
    """
    try:
        if data_args.image_folder.startswith("s3://"):
            # Parse bucket and prefix
            bucket_name = data_args.image_folder[5:]  # remove 's3://'
            s3_config = json.load(open("work_dirs/s3.json", "r"))
            s3_client = boto3.client(
                service_name='s3',
                endpoint_url=s3_config['endpoint_url'],
                aws_access_key_id=s3_config['aws_access_key_id'],
                aws_secret_access_key=s3_config['aws_secret_access_key'],
            )

            for image_file in image_files:
                print(f"Fetching from S3: bucket={bucket_name}, key={image_file}")
                response = s3_client.get_object(Bucket=bucket_name, Key=image_file)
                image_data = response['Body'].read()
                image = Image.open(BytesIO(image_data)).convert("RGB")
                print(f"✅ Loaded image {image_file} with size {image.size}")

        else:
            for image_file in image_files:
                path = os.path.join(data_args.image_folder, image_file)
                print(f"Opening local file: {path}")
                image = Image.open(path).convert("RGB")
                print(f"✅ Loaded local image {image_file} with size {image.size}")

    except Exception as e:
        print("❌ Error while testing S3/local image loading:")
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    class DummyArgs:
        image_folder = "s3://image"  # or "./local_images"

    # Example image keys (adjust based on your JSON data)
    test_images = ["cc12m/part1/2206336_2434534077.jpg"]

    test_s3_connection(DummyArgs(), test_images)
