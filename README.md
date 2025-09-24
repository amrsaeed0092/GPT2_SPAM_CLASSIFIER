# 📌 GPT-2 From Scratch — Spam Classifier & Text Generator  

##  Project Overview  
This project implements **GPT-2 from scratch** using PyTorch, including all the core building blocks:  
- 🧩 Token embeddings  
- 📐 Positional encoding layer  
- 🔄 Transformer blocks (multi-head attention + feed-forward networks)  
- ⚖️ Layer normalization and residual connections  
- 📝 Output classification/decoding head  

After building the architecture, I:  
1. **Downloaded the pretrained GPT-2 weights** released by OpenAI.  
2. **Fine-tuned the model** for two tasks:  
   - **Text generation** (creative continuation of input prompts).  
   - **Spam classification** (binary classification: SPAM vs HAM).  

The model achieves **100% accuracy** on the spam classification task, showing its ability to both generate coherent text and act as a robust classifier.  

## ✨ Features  

- ✅ **GPT-2 architecture** fully re-implemented from scratch  
- ✅ Supports **text generation** with autoregressive decoding  
- ✅ Supports **spam classification** with fine-tuning  
- ✅ Integrated with **pretrained OpenAI GPT-2 weights**  
- ✅ Easy to **retrain and extend** to new tasks  

## 📂 Repository Structure  

```plaintext
├── modules/                   # Core modules (attention, transformer blocks, etc.)  
├── config/                    # Configurations and hyperparameters  
├── data/                      # Dataset files (if applicable)  
├── models                     # Folder to save the trained models 
├── Figures                    # Folder to save the plotted figures 
├── train.py                   # Script to retrain the GPT-2 model 
├── main.py                    # the start point to the GPT-2 model  
├── SpamClassifierTester.ipynb # Notebook to test the fine-tuned spam classifier  
└── README.md                  # Project documentation  
