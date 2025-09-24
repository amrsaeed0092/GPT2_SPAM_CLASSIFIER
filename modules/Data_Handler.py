import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from config import DataConfig as dcnfg
import tiktoken


'''
THREE STEPS ARE PERFORMED IN THIS STEP:
Step 1: Pre-tokenize texts
Step 2: Truncate sequences if they are longer than max_length
Step 3: Pad sequences to the longest sequence

THIS CLASS RETURN THE ENCODED TEXT AND LABEL TENSORS 
'''
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    

def createDataLoaders():
    #set the dataset path
    data_file_path = Path(dcnfg.extracted_path) 
    
    torch.manual_seed(123)

    # Initialize the BPE tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #INITIALIZE DATASETS
    train_dataset = SpamDataset(
        csv_file=f"{data_file_path}/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=f"{data_file_path}/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=f"{data_file_path}/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    #INITIALIZE DATALOADERS
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dcnfg.batch_size,
        shuffle=True,
        num_workers=dcnfg.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=dcnfg.batch_size,
        num_workers=dcnfg.num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=dcnfg.batch_size,
        num_workers=dcnfg.num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader