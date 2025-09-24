import os
import sys
from pathlib import Path
#from config import DataConfig as dcnfg
from modules import data_downloader as dowload
from modules.exception import CustomException
import pandas as pd
from modules.logger import logging, get_log_file_name

class DataPreprocessor:
    def __init__(self, dcnfg):
        self.DataFolder = dcnfg.extracted_path
        self.DatasetName = dcnfg.dataset_name
        self.data_file_path = Path(dcnfg.extracted_path) / dcnfg.dataset_name

        #DOWNLOAD THE DATASET
        dowload.download_and_unzip_spam_data(dcnfg.url, dcnfg.zip_path, dcnfg.extracted_path, self.data_file_path)

        #READ THE DATASET INTO A DATAFRAME
        self.input_df =  pd.read_csv(self.data_file_path, sep="\t", header=None, names= ["Label", "Text"])

        #CREATE A BALANCED DATASET
        self.balanced_df = self.createBalanceDataSet()

        # PERFORM LABEL ENCODING 
        self.labelEncoder()

        #TRAIN/VALIDATION/TEST SPILIT
        self.train_df, self.validation_df, self.test_df = self.trainTestSpilit(0.7, 0.1) # Test size is implied to be 0.2 as the remainder
        

    def createBalanceDataSet(self):
        # Count the instances of "spam"
        num_spam = self.input_df[self.input_df["Label"] == "spam"].shape[0]
        
        # Randomly sample "ham" instances to match the number of "spam" instances
        ham_subset = self.input_df[self.input_df["Label"] == "ham"].sample(num_spam, random_state=123)
        
        # Combine ham "subset" with "spam"
        balanced_df = pd.concat([ham_subset, self.input_df[self.input_df["Label"] == "spam"]])

        return balanced_df

    #Perform Label encoding
    # 0 refers to not spam (ham)
    # 1 refers to spam
    def labelEncoder(self):
        self.balanced_df["Label"] = self.balanced_df["Label"].map({"ham":0, "spam": 1} )


    def trainTestSpilit(self, train_frac, validation_frac):
         # Shuffle the entire DataFrame
         df = self.balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

         # Calculate split indices
         train_end = int(len(df) * train_frac)
         validation_end = train_end + int(len(df) * validation_frac)

         # Split the DataFrame
         train_df = df[:train_end]
         validation_df = df[train_end:validation_end]
         test_df = df[validation_end:]
        
         # save data files as CSV files
         try:
            train_df.to_csv(f"{self.DataFolder}/train.csv", index=None)
            validation_df.to_csv(f"{self.DataFolder}/validation.csv", index=None)
            test_df.to_csv(f"{self.DataFolder}/test.csv", index=None)
            logging.info("training, testing and validation datasets created in {self.data_file_path}")
         except Exception as e:
            raise CustomException (e, sys)
         
         return train_df, validation_df, test_df
    