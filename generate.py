import os
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from Commons import Commons
from LyricsPreprocessor import LyricsPreprocessor

# # CUDA_LAUNCH_BLOCKING=1

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
generation_count = 3
max_length = 1024


tokenizer.add_special_tokens(Commons.special_tokens)

force_cpu = False
# force_cpu = True # debug on CPU

# Make sure the model is on the correct device (GPU if available)
device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# prompt = f"{LyricsPreprocessor.MARKER_SONG_NAME_START}"
# title = "Unleash the madness"
# title = "Run away"
title = "Mustakrakish"

prompt = f"{LyricsPreprocessor.bos_token}{LyricsPreprocessor.MARKER_SONG_NAME_START}{title}{LyricsPreprocessor.MARKER_SONG_NAME_END}{LyricsPreprocessor.MARKER_END_OF_LINE}"

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

token_song_name_start = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_START)
token_song_name_end = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_END)
token_song_start = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_START)
token_song_end = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_END)

PAD_TOKEN_id = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.PAD_TOKEN)
bos_token_id = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.bos_token)
eos_token_id = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.eos_token)

def postprocess(lyrics):
    lyrics = lyrics.replace(LyricsPreprocessor.MARKER_END_OF_LINE, '\n')
    lyrics = lyrics.replace(LyricsPreprocessor.eos_token, '')
    lyrics = lyrics.replace(LyricsPreprocessor.bos_token, '')
    lyrics = lyrics.replace(LyricsPreprocessor.MARKER_SONG_NAME_START, '')
    lyrics = lyrics.replace(LyricsPreprocessor.MARKER_SONG_NAME_END, '\n')
    return lyrics

def get_timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

run_output_dir = f"{output_dir}/run-{get_timestamp()}"

os.makedirs(run_output_dir, exist_ok=True)
print(f"Created {run_output_dir}")

with torch.no_grad():
    for i in range(generation_count):
        filename = f"{run_output_dir}/song-{get_timestamp()}.txt"
        print(f"Generating {i+1}, will be saved to {filename}")

        # Generate text
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,                          # Maximum length of generated text
            num_return_sequences=1,                  # Number of generated texts
            no_repeat_ngram_size=2,                  # Avoid repeating n-grams
            top_k=50,                                # Limits sampling to the top k most probable next tokens
            temperature=0.9,                         # Controls randomness in output (lower = less random)
            top_p=0.95,                              # Nucleus sampling (keeps cumulative probability <= top_p)
            do_sample=True,                          # Enable sampling (instead of greedy decoding)

            # pad_token_id=tokenizer.eos_token_id,
            # forced_bos_token_id=token_song_name_start,
            # forced_eos_token_id=token_song_end,
            pad_token_id=PAD_TOKEN_id,
            forced_bos_token_id=bos_token_id,
            forced_eos_token_id=eos_token_id,
            num_beams=10,
        )
        # Decode and print the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        torch.cuda.empty_cache()

        postprocessed_lyrics = postprocess(generated_text)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(postprocessed_lyrics)

        del generated_text
        del output
