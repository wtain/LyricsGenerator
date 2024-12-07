import random

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Make sure the model is on the correct device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# model.pad_token_id = model.eos_token_id

# Define a prompt to start the text generation
# prompt = "In the beginning, the world was"
# prompt = "That darkness"
prompt = "i try my hardest to be stronger than"

# dataset_path = "dataset.txt"  # Replace with your dataset path
# dataset = load_dataset("text", data_files={"train": dataset_path})
#
# prompt = random.choice(dataset["train"])['text']

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    input_ids=input_ids,                    # Input prompt
    max_length=200,                          # Maximum length of generated text
    num_return_sequences=1,                  # Number of generated texts
    no_repeat_ngram_size=2,                  # Avoid repeating n-grams
    temperature=0.9,                         # Controls randomness in output (lower = less random)
    top_k=50,                                # Limits sampling to the top k most probable next tokens
    top_p=0.95,                              # Nucleus sampling (keeps cumulative probability <= top_p)
    do_sample=True,                          # Enable sampling (instead of greedy decoding)
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
