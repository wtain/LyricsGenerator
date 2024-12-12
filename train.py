import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

from Commons import Commons
from LyricsPreprocessor import LyricsPreprocessor


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Load your dataset
    dataset_path = "dataset.txt"  # Replace with your dataset path
    dataset = load_dataset("text", data_files={"train": dataset_path})

    model_name = "gpt2"  # You can also try "gpt2-medium" or other variants
    # model_name = "gpt2-large"  # You can also try "gpt2-medium" or other variants
    out_model = "./fine_tuned_gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(Commons.special_tokens)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Move the model to the GPU
    model = model.to("cuda")

    print("Device used for training:", model.device)  # Should print something like 'cuda:0'

    # Tokenize the dataset
    def tokenize_function(examples):
        # Create input_ids and labels
        # The labels are the same as input_ids
        encodings = tokenizer(examples["text"],
                              truncation=True,
                              padding="max_length",
                              max_length=128,
                              return_tensors="pt",  # Returns PyTorch tensors
                              return_attention_mask=True
        )
        encodings["labels"] = encodings["input_ids"].copy()  # Set labels to input_ids
        return encodings

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Define training arguments
    num_train_epochs = 3
    training_args = TrainingArguments(
        output_dir="./results",        # Directory to save checkpoints
        overwrite_output_dir=True,    # Overwrite the output directory if it exists
        num_train_epochs=num_train_epochs,           # Number of epochs
        per_device_train_batch_size=8,  # Batch size per device
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 8 = 32
        save_steps=500,               # Save checkpoint every 500 steps
        save_total_limit=2,           # Keep only the last 2 checkpoints
        logging_dir="./logs",         # Directory for logs
        logging_steps=50,             # Log every 50 steps
        eval_strategy="no",        # Evaluate the model during training
        eval_steps=500,               # Evaluation frequency
        learning_rate=5e-5,           # Learning rate
        warmup_steps=500,             # Warmup steps for learning rate
        fp16=True,                    # Use mixed precision training (if supported by your hardware)
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(out_model)
    tokenizer.save_pretrained(out_model)

if __name__ == "__main__":
    main()