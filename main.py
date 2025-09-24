
import matplotlib.pyplot as plt
from modules.DataPreprocessor import DataPreprocessor
from config import MODEL_CONFIG, DataConfig, AppConfig
from modules.logger import logging, get_log_file_name
from modules.Data_Handler import createDataLoaders
from modules.GPT2_Model import GPTModel , generate_text_simple
import modules.text_generation  as text_gen
from modules.gpt2_weight_dowloader import download_and_load_gpt2, load_weights_into_gpt
from SpamTrainer import SpamTrainer
import os
import torch
import time 

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

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


def train(model, config, train_loader, val_loader):
    
    start_time = time.time()
    logging.info(f"Training Started at {start_time}")
    
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["Learning_Rate"], weight_decay=config["Weight_Decay"])

    Trainer = SpamTrainer(
        model=model, 
        optimizer = optimizer,
        train_loader = train_loader, 
        val_loader = val_loader, 
        device = device,
        num_epochs=config["Epochs"],
        eval_freq=config["Evaluation_Freq"],
        eval_iter = config["Evaluation_Iteration"]
        )
    train_losses, val_losses, train_accs, val_accs, examples_seen = Trainer.train_val()

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logging.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return Trainer.model, train_losses, val_losses, train_accs, val_accs, examples_seen 

'''
THIS FUNCTION IS USED TO SAVE THE TRAINING ACCURACY AND LOSS 
'''
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{AppConfig.FIGURE_DIR}/{label}-plot.pdf")
    plt.show()


def main():
    '''
    STEP 1: 
    1. DOWNLOAD AND PREPROCESS THE DATASET
    2. TRAIN/TEST/VALIDATION SPLIT
    '''
    t = DataPreprocessor(DataConfig)

    '''
    STEP 2: CREATE ITERABLE DATASET OBJECTS USING PYTORCH DATASET AND DATALOADER CLASSES
    '''
    train_loader, val_loader, test_loader = createDataLoaders()

   
    '''
    STEP 3: DOWNLOAD THE GPT2 PRETRAINED WIEGHTS AND FILES FROM https://openaipublic.blob.core.windows.net/gpt-2/models
    '''
    settings, params = download_and_load_gpt2(model_size=AppConfig.GPT2_MODEL_SIZE, models_dir=AppConfig.GPT2_WEIGHT_MODEL_DIR)

    '''
    Use the following function to generate new tokens using the pretrained model
    '''
    #evaluate_pretrained_weights(settings, params)
    
    '''
    STEP 4: FINE TUNE THE MODEL (TRAIN THE MODEL)
    '''
    MODEL_CONFIG.GPT_CONFIG_124M.update(MODEL_CONFIG.GPT_MODEL_CONFIGS[MODEL_CONFIG.MODEL_NAME_TO_USE])
    model_size = MODEL_CONFIG.MODEL_NAME_TO_USE.split(" ")[-1].lstrip("(").rstrip(")")

    model = GPTModel(MODEL_CONFIG.GPT_CONFIG_124M)
    model = load_weights_into_gpt(model, params)
    
    # To get the model ready for classification-finetuning, we first freeze the model, meaning that we make all layers non-trainable
    for param in model.parameters():
        param.requires_grad = False

    #Add the classification head
    model.out_head = torch.nn.Linear(in_features=MODEL_CONFIG.GPT_CONFIG_124M["emb_dim"], 
                                     out_features=MODEL_CONFIG.TRAINING_CONFIG["num_classes"])
    
    #Only fine tune the last transformer block and the final LayerNorm module, which connects this block to the output layer
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    # TRAIN (FINE TUNE) THE MODEL
    model, train_losses, val_losses, train_accs, val_accs, examples_seen  = train(model = model.to(device), config=MODEL_CONFIG.TRAINING_CONFIG, train_loader = train_loader, val_loader= val_loader)
    
    #DRAW THE LOSS VALUES
    epochs_tensor = torch.linspace(0, MODEL_CONFIG.TRAINING_CONFIG["Epochs"], len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


    #DRAW THE ACCURACY VALUES
    epochs_tensor = torch.linspace(0, MODEL_CONFIG.TRAINING_CONFIG["Epochs"], len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    #SAVE THE TRAINED MODEL
    saveModels(model)

#Save the fine tuned models for future use
def saveModels(model):
    torch.save(model.state_dict(), f"{AppConfig.SPAM_TRAINED_MODEL_DIR}/Spam Classifier.pth")    

if __name__ == "__main__":
    main()
