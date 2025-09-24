
from modules.DataPreprocessor import DataPreprocessor
from config import MODEL_CONFIG, DataConfig, AppConfig
from modules.Data_Handler import createDataLoaders
from modules.GPT2_Model import GPTModel , generate_text_simple
import modules.text_generation  as text_gen
from modules.gpt2_weight_dowloader import download_and_load_gpt2, load_weights_into_gpt
import os
import torch
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_pretrained_weights(settings, params):
    MODEL_CONFIG.GPT_CONFIG_124M.update(MODEL_CONFIG.GPT_MODEL_CONFIGS[MODEL_CONFIG.MODEL_NAME_TO_USE])
    model_size = MODEL_CONFIG.MODEL_NAME_TO_USE.split(" ")[-1].lstrip("(").rstrip(")")

    model = GPTModel(MODEL_CONFIG.GPT_CONFIG_124M)
    model = load_weights_into_gpt(model, params)
    model.to(device)
    model.eval();
    #test text generation function without training the model
    Output = text_gen.example(model, max_new_tokens=15, example_text= "Every effort moves you")
    print(Output)


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


    '''
    DOWNLOAD THE GPT2 PRETRAINED WIEGHTS AND FILES FROM https://openaipublic.blob.core.windows.net/gpt-2/models
    '''
    settings, params = download_and_load_gpt2(model_size=AppConfig.GPT2_MODEL_SIZE, models_dir=AppConfig.GPT2_WEIGHT_MODEL_DIR)
    print(params)

    evaluate_pretrained_weights(settings, params)
    
    

if __name__ == "__main__":
    main()
