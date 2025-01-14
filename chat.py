import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
output_model = "tinyllama-XOR"  # Change to your model path
model = AutoModelForCausalLM.from_pretrained(output_model)
tokenizer = AutoTokenizer.from_pretrained(output_model)

# Ensure the tokenizer's pad token is set correctly
tokenizer.pad_token = tokenizer.eos_token

# Function to generate response
def generate_response(input_text):
    # Format the input as expected by the model
    input_ids = tokenizer(f"<|user|>\n{input_text}</s>\n", return_tensors="pt").input_ids

    # Generate the response
    output_ids = model.generate(input_ids, max_length=100)

    # Decode the output tokens to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response
