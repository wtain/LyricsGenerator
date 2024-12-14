import os
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from Commons import Commons
from LyricsPreprocessor import LyricsPreprocessor

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
generation_count = 20


tokenizer.add_special_tokens(Commons.special_tokens)


# Make sure the model is on the correct device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.resize_token_embeddings(len(tokenizer))
model.to(device)

prompt = f"{LyricsPreprocessor.MARKER_SONG_NAME_START}"

# Tokenize the input prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

token_song_name_start = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_START)
token_song_name_end = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_NAME_END)
token_song_start = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_START)
token_song_end = tokenizer.convert_tokens_to_ids(LyricsPreprocessor.MARKER_SONG_END)

def postprocess(lyrics):
    lyrics = lyrics.replace(LyricsPreprocessor.MARKER_END_OF_LINE, '\n')
    lyrics = lyrics.replace('<|endoftext|>', '')
    return lyrics

for i in range(generation_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H%M%S')
    filename = f"{output_dir}/{timestamp}.txt"

    # Generate text
    output = model.generate(
        input_ids=input_ids,
        max_length=1500,                          # Maximum length of generated text
        num_return_sequences=1,                  # Number of generated texts
        no_repeat_ngram_size=2,                  # Avoid repeating n-grams
        top_k=50,                                # Limits sampling to the top k most probable next tokens
        temperature=0.9,                         # Controls randomness in output (lower = less random)
        top_p=0.95,                              # Nucleus sampling (keeps cumulative probability <= top_p)
        do_sample=True,                          # Enable sampling (instead of greedy decoding)

        pad_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=token_song_name_start,
        forced_eos_token_id=token_song_end,
        num_beams=10,
    )
    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    postprocessed_lyrics = postprocess(generated_text)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(postprocessed_lyrics)
