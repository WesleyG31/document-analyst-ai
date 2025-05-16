import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils import read_yaml


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config=config['data_ingestion']
        self.bucket_name=self.config['bucket_name']
        self.file_names= self.config['bucket_file']

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"######### Data ingestion started ######")

    def download_csv_from_gcp(self):
        try: 
            logger.info("Downloading CSV files from GCP bucket")

            client= storage.Client()
            bucket=client.bucket(self.bucket_name)
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    
