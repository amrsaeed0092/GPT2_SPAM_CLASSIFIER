import urllib.request
import ssl
import zipfile
import os, sys
from pathlib import Path
from config import DataConfig as dcnfg
from modules.exception import CustomException
from modules.logger import logging, get_log_file_name

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    
    try:
        if data_file_path.exists():
            logging.info(f"{data_file_path} already exists. Skipping download and extraction.")
            return

        # Create an unverified SSL context
        ssl_context = ssl._create_unverified_context()

        # Downloading the file
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

        # Unzipping the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)

        # Add .tsv file extension
        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        logging.info(f"File downloaded and saved as {data_file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e


