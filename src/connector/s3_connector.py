import os
from pathlib import Path
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel
from config.setting import settings


class S3FileMetadata(BaseModel):
    bucket: str
    key: str
    size_bytes: int
    last_modified: str


class S3Connector:
    """Connects to AWS S3 bucket to list and download documents for ingestion."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        self.region = region_name or os.getenv("AWS_REGION", "us-east-1")
        
        # Initializes AWS S3 Client using env vars or default AWS credentials
        self.s3_client = boto3.client("s3", region_name=self.region)

    def list_objects(self, prefix: str = "") -> List[S3FileMetadata]:
        """Lists files available inside the target S3 bucket prefix/folder."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    # Skip folder objects
                    if not obj["Key"].endswith("/"):
                        files.append(
                            S3FileMetadata(
                                bucket=self.bucket_name,
                                key=obj["Key"],
                                size_bytes=obj["Size"],
                                last_modified=str(obj["LastModified"])
                            )
                        )
            return files
        except ClientError as e:
            print(f"Error connecting to S3 Bucket: {e}")
            return []

    def download_file(self, s3_key: str, local_destination: Path) -> Path:
        """Downloads a specific S3 object to local disk (e.g. data/raw/)."""
        local_path = Path(local_destination) / Path(s3_key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            print(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} -> {local_path}")
            return local_path
        except ClientError as e:
            raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")


if __name__ == "__main__":
    print("S3Connector initialized.")