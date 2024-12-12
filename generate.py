import os
import random
from datetime import datetime, date

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from LyricsPreprocessor import LyricsPreprocessor

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
generation_count = 20

# copied from train.py
special_tokens = {
    "additional_special_tokens": [
        LyricsPreprocessor.MARKER_END_OF_LINE,
        LyricsPreprocessor.MARKER_SONG_NAME_START,
        LyricsPreprocessor.MARKER_SONG_NAME_END,
        LyricsPreprocessor.MARKER_SONG_START,
        LyricsPreprocessor.MARKER_SONG_END,
    ]
}
tokenizer.add_special_tokens(special_tokens)


# Make sure the model is on the correct device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda:1")  # Use GPU 1
model.resize_token_embeddings(len(tokenizer))
model.to(device)
# model.pad_token_id = model.eos_token_id

# Define a prompt to start the text generation
# prompt = "In the beginning, the world was"
# prompt = "That darkness"
# prompt = "i try my hardest to be stronger than"


# prompt = f"{LyricsPreprocessor.MARKER_SONG_START}{LyricsPreprocessor.MARKER_SONG_NAME_START}"
prompt = f"{LyricsPreprocessor.MARKER_SONG_NAME_START}"

# dataset_path = "dataset.txt"  # Replace with your dataset path
# dataset = load_dataset("text", data_files={"train": dataset_path})
#
# prompt = random.choice(dataset["train"])['text']

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
# input_ids = tokenizer["input_ids"].to(device)
# attention_mask = tokenizer["attention_mask"].to(device)

# forced_tokens = tokenizer.convert_tokens_to_ids([
#     LyricsPreprocessor.MARKER_SONG_NAME_END,
#     LyricsPreprocessor.MARKER_SONG_END,
# ])

id0 = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_START)
id1 = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_END)
id2 = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_START)
id3 = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_END)
forced_tokens = [[id1], [id2], [id3]]

# forced_tokens = tokenizer.encode([
#     LyricsPreprocessor.MARKER_SONG_NAME_END,
#     LyricsPreprocessor.MARKER_SONG_END,
# ], return_tensors="pt").to(device)

# forced_tokens = tokenizer.convert_tokens_to_ids([
#     # tokenizer([LyricsPreprocessor.MARKER_SONG_NAME_END], add_prefix_space=True, add_special_tokens=False).input_ids,
#     # tokenizer([LyricsPreprocessor.MARKER_SONG_END], add_prefix_space=True, add_special_tokens=False).input_ids,
#
#     [tokenizer.encode(LyricsPreprocessor.MARKER_SONG_NAME_END, add_prefix_space=True, add_special_tokens=False, return_tensors="pt").to(device)],
#     [tokenizer.encode(LyricsPreprocessor.MARKER_SONG_END, add_prefix_space=True, add_special_tokens=False, return_tensors="pt").to(device)],
# ])



def postprocess(lyrics):
    lyrics = lyrics.replace(LyricsPreprocessor.MARKER_END_OF_LINE, '\n')
    lyrics = lyrics.replace('<|endoftext|>', '')
    return lyrics

for i in range(generation_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H%M%S')
    filename = f"{output_dir}/{timestamp}.txt"

    # Generate text
    output = model.generate(
        input_ids=input_ids,                    # Input prompt
        max_length=1500,                          # Maximum length of generated text
        num_return_sequences=1,                  # Number of generated texts
        no_repeat_ngram_size=2,                  # Avoid repeating n-grams
        top_k=50,                                # Limits sampling to the top k most probable next tokens
        temperature=0.9,                         # Controls randomness in output (lower = less random)
        top_p=0.95,                              # Nucleus sampling (keeps cumulative probability <= top_p)
        do_sample=True,                          # Enable sampling (instead of greedy decoding)

        pad_token_id=tokenizer.eos_token_id,
        # force_words_ids=forced_tokens,
        forced_bos_token_id=id0, # song name start
        forced_eos_token_id=id3, # song end
        # attention_mask=attention_mask,
        num_beams=10,
        # do_sample=False,
        # diversity_penalty=0.5,  # Encourage diversity among beams

        # do_sample=True,  # Enable sampling
        # top_k=50,  # Combine top-k sampling with beams
        # top_p=0.9,  # Use nucleus sampling for added diversity
    )
    # Decode and print the generated text
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)
    print('\n')
    postprocessed_lyrics = postprocess(generated_text)
    print(postprocessed_lyrics)
    with open(filename, "w") as file:
        file.write(postprocessed_lyrics)
