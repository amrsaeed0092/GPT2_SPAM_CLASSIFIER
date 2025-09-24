
from modules.DataPreprocessor import DataPreprocessor
from config import DataConfig, MODEL_CONFIG
from modules.Data_Handler import createDataLoaders
from modules.GPT2_Model import GPTModel , generate_text_simple
import modules.text_generation  as text_gen


def main():
    '''
    1. DOWNLOAD AND PREPROCESS THE DATASET
    2. TRAIN/TEST/VALIDATION SPLIT
    '''
    #t = DataPreprocessor(DataConfig)

    '''
    CREATE ITERABLE DATASET OBJECTS USING PYTORCH DATASET AND DATALOADER CLASSES
    '''
    #train_loader, val_loader, test_loader = createDataLoaders()

    '''
    #TEST THE RESULTS
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
    '''

    #test text generation function without training the model
    Output = text_gen.example(max_new_tokens=12, example_text= "Please let me ")
    print(Output)

if __name__ == "__main__":
    main()
