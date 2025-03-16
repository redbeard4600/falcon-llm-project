import argparse
from falcon.inference import load_model, generate_text, load_pipeline_model

def main():
    parser = argparse.ArgumentParser(description="Falcon 7B CLI")
    parser.add_argument("--model", type=str, default="tiiuae/falcon-7b", help="Model ID")
    parser.add_argument("--quantization", type=str, choices=[None, "4bit", "8bit"], default="4bit", help="Quantization type")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    
    args = parser.parse_args()
    
    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, args.quantization)
    
    print("Generating text...")
    output = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        args.max_length,
        args.temperature,
        args.top_p
    )
    
    print("\nGenerated text:")
    print(output)

if __name__ == "__main__":
    main()