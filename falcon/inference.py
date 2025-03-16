# falcon/inference.py
from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

@lru_cache(maxsize=1)
def load_model(model_id, quantization=None):
    """
    Load model with caching for faster subsequent loads.
    Args:
        model_id (str): HuggingFace model ID
        quantization (str, optional): Quantization type ('4bit', '8bit', or None)
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure model loading based on quantization parameter
    if quantization == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif quantization == '8bit':
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """
    Generate text using the loaded model.
    Args:
        model: The loaded language model
        tokenizer: The loaded tokenizer
        prompt (str): Input text to generate from
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Controls diversity via nucleus sampling
    Returns:
        str: Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def batch_generate(model, tokenizer, prompts, **kwargs):
    """
    Generate text for multiple prompts.
    Args:
        model: The loaded language model
        tokenizer: The loaded tokenizer
        prompts (list): List of input prompts
        **kwargs: Additional parameters for generate_text
    Returns:
        list: List of generated texts
    """
    return [generate_text(model, tokenizer, prompt, **kwargs) for prompt in prompts]

# Add this function to falcon/inference.py
def load_model_cpu(model_id):
    """
    Load model on CPU without quantization
    Args:
        model_id (str): HuggingFace model ID
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32  # Use float32 for CPU
    )
    return model, tokenizer

def load_pipeline_model(model_id, quantization=None):
    """
    Load model as a pipeline for easier text generation.
    Args:
        model_id (str): HuggingFace model ID
        quantization (str, optional): Quantization type ('4bit', '8bit', or None)
    Returns:
        pipeline: Hugging Face text generation pipeline
    """
    if quantization:
        model, tokenizer = load_model(model_id, quantization)
    else:
        model, tokenizer = load_model_cpu(model_id)
        
    from transformers import pipeline
    return pipeline("text-generation", model=model, tokenizer=tokenizer)