from transformers import TrainingArguments, Trainer
import os

def fine_tune_model(model, tokenizer, dataset, output_dir="./fine_tuned_model", 
                    epochs=1, learning_rate=2e-5, batch_size=4):
    """
    Fine-tune a loaded model on a custom dataset.
    Args:
        model: The loaded model to fine-tune
        tokenizer: The tokenizer for the model
        dataset: Dataset to fine-tune on (HuggingFace dataset format)
        output_dir (str): Directory to save the fine-tuned model
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for training
        batch_size (int): Batch size for training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def prepare_dataset(texts, tokenizer, max_length=512):
    """
    Prepares a simple dataset from text samples.
    Args:
        texts (list): List of text samples
        tokenizer: The model tokenizer
        max_length (int): Maximum sequence length
    Returns:
        dict: Dataset dictionary compatible with HuggingFace Trainer
    """
    encodings = tokenizer(texts, truncation=True, padding="max_length", 
                         max_length=max_length, return_tensors="pt")
    
    class SimpleDataset:
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item
            
        def __len__(self):
            return len(self.encodings["input_ids"])
    
    return SimpleDataset(encodings)