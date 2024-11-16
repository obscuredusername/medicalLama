# Load the required packages.
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import pandas as pd
import json

dataset_path = "commands_dataset.json"  # Path to your JSON dataset
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_model = "tinyllama-XOR"

def formatted_train(input, command) -> str:
    return f"<|user|>\n{input}</s>\n<|assistant|>\n{command}</s>"

# Prepare train data from your JSON file
def prepare_train_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    data_df = pd.DataFrame(data)

    # Assumes 'input' and 'command' are the relevant columns in the JSON
    data_df["text"] = data_df.apply(lambda x: formatted_train(x["input"], x["command"]), axis=1)

    # Convert to Hugging Face Dataset format
    hf_dataset = Dataset.from_pandas(data_df)
    return hf_dataset

# Prepare medical chatbot data
def prepare_medical_chatbot_data():
    medical_data = load_dataset("ruslanmv/ai-medical-chatbot")["train"]
    
    # Format the medical chatbot data
    medical_df = pd.DataFrame({
        "input": medical_data["Description"],
        "command": medical_data["Doctor"]  # Change this to the doctor's response
    })
    
    # Format it similarly
    medical_df["text"] = medical_df.apply(lambda x: formatted_train(x["input"], x["command"]), axis=1)
    
    # Convert to Hugging Face Dataset format
    medical_hf_dataset = Dataset.from_pandas(medical_df)
    return medical_hf_dataset

# Load your datasets
data = prepare_train_data(dataset_path)
medical_data = prepare_medical_chatbot_data()

# Combine the datasets using concatenate_datasets
combined_data = concatenate_datasets([data, medical_data])

# Load the model and tokenizer
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

# PEFT configuration
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=1,  # Decreased batch size
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
    optim="paged_adamw_32bit",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=5,
    max_steps=250,
    fp16=True,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=combined_data,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_arguments,
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024
)

# Start training
trainer.train()

# Save the model after training
trainer.save_model(output_model)
