import tiktoken
import torch
from config import MODEL_CONFIG
from modules.GPT2_Model import GPTModel , generate_text_simple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def example(model, max_new_tokens, example_text = "Hello, I am"):
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = example_text
    encoded_tensor = text_to_token_ids(start_context, tokenizer)

    token_ids = generate_text_simple(
    model=model,
    idx=encoded_tensor.to(device),
    max_new_tokens=max_new_tokens,
    context_size=MODEL_CONFIG.GPT_CONFIG_124M["context_length"]
    )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    return ({"Start Context": start_context
            , "Generated Context": decoded_text})


