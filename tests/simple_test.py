import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly (without quantization)
model_path = "tiiuae/falcon-7b-instruct"
print(f"Loading model {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "Explain quantum computing in simple terms:"
print(f"Generating with prompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        do_sample=True,
        temperature=0.7,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nResponse:")
print(response)