import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, Dataset

def get_device(preferred_device: str = "device_that_not_exist"):
    """
    Determines the most suitable device for PyTorch computations.

    Args:
        preferred_device (str): The preferred device to use (e.g., "cuda", "mps", "hpu").
            If not available, the next available device in the list will be used.

    Returns:
        torch.device: The selected device.
    """

    device_list = ["cuda", "mps", "hpu", "cpu"]

    # Prioritize the preferred device if available
    if preferred_device in device_list:
        device_list.insert(0, preferred_device)

    for device in device_list:
        try:
            # Check if the device is available
            if device == "cpu":
                # CPU is always available
                return torch.device(device)
            elif hasattr(torch, device) and hasattr(getattr(torch, device), "is_available"):
                if getattr(torch, device).is_available():
                    return torch.device(device)
            elif hasattr(torch.backends, device) and hasattr(getattr(torch.backends, device), "is_built"):
                if getattr(torch.backends, device).is_built():
                    return torch.device(device)
        except ValueError:
            pass

    raise RuntimeError("No compatible device found.")


def tokenize_function(examples, tokenizer, max_length):
    #texts = [text + tokenizer.eos_token for text in examples["text"]]
    texts = examples["text"]
    tokenized = tokenizer(texts, truncation=True, padding=True)
    return tokenized
    #return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def main():
    device = get_device("mps")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Set up configuration
    max_length = 16
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=16,
        max_position_embeddings=len(tokenizer),
    )

    # Initialize model
    model = LlamaForCausalLM(config).to(device)

    example_data = [
        "What's your name? My name is wilson",
        "Who is the author of \"Pride and Prejudice\"??",
        "What is the capital of Australia?",
        "When did the French Revolution begin?",
        "What is the largest planet in our solar system?",
        # Add more examples here...
    ]


    # Load and prepare dataset
    #raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    #raw_dataset = load_dataset("liswei/wikipedia-zhtw-dedup", split="train")
    raw_dataset = Dataset.from_dict({"text": example_data})
    inp_data = tokenizer.decode(tokenizer.encode(raw_dataset['text'][0], return_tensors='pt')[0][:max_length])
    print("Example input data:", inp_data)
    
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )


    # Define training arguments
    train_epochs = 4800
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=train_epochs,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=(train_epochs / 10),
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./llama_model")
    tokenizer.save_pretrained("./llama_model")
    # Example of using the trained model
    input_text = "今"
    #inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set up generation config
    generation_config = GenerationConfig(
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Generate text
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()