# In falcon/app.py, update the caching annotation
import streamlit as st
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model, tokenizer, prompt, max_length, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
# Edit falcon/app.py
@st.cache_resource
def load_model(model_id, quantization):
    if quantization == "4bit":
        return AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_4bit=True,
            device_map="auto"
        ), AutoTokenizer.from_pretrained(model_id)
    elif quantization == "8bit":
        return AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_8bit=True,
            device_map="auto"
        ), AutoTokenizer.from_pretrained(model_id)

@st.cache_resource
def load_model_cpu(model_id):
    return AutoModelForCausalLM.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id)

@st.cache_resource
def get_model(model_id, quantization):
    if quantization:
        return load_model(model_id, quantization)
    else:
        return load_model_cpu(model_id)
        return load_model_cpu(model_id)

def main():
    st.title("Falcon 7B Text Generation")
    
    # Sidebar for model settings
    st.sidebar.header("Model Settings")
    model_id = st.sidebar.text_input("Model ID", "tiiuae/falcon-7b")
    quantization = st.sidebar.selectbox("Quantization", 
                                       [None, "8bit", "4bit"], 
                                       index=2)
    
    # Load model button
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model, tokenizer = get_model(model_id, quantization)
            st.session_state['model'] = model
            st.session_state['tokenizer'] = tokenizer
            st.sidebar.success("Model loaded successfully!")
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", 10, 500, 100)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9)
    
    # Text input and generation
    prompt = st.text_area("Enter your prompt:", height=150)
    
    if st.button("Generate"):
        if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
            st.error("Please load the model first!")
        elif not prompt:
            st.error("Please enter a prompt!")
        else:
            with st.spinner("Generating..."):
                try:
                    generated_text = generate_text(
                        st.session_state['model'],
                        st.session_state['tokenizer'],
                        prompt,
                        max_length,
                        temperature,
                        top_p
                    )
                    st.success("Generation complete!")
                    st.markdown("### Generated Text")
                    st.write(generated_text)
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()