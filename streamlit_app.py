# streamlit_app.py
import streamlit as st
import torch
import sys
import os

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import load_model, generate_text, tokenizer
from monitoring import monitor
import pandas as pd
import plotly.express as px

def main():
    st.set_page_config(page_title="Falcon 7B Interface", layout="wide")
    
    st.title("Falcon 7B Text Generation")
    
    # Sidebar with model parameters
    st.sidebar.header("Model Parameters")
    
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                                   help="Higher values make output more random, lower more deterministic")
    
    max_length = st.sidebar.slider("Maximum Length", 50, 500, 100, 10, 
                                  help="Maximum length of generated text")
    
    top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9, 0.05, 
                             help="Nucleus sampling parameter")
    
    # Prompt templates
    st.sidebar.header("Prompt Templates")
    template = st.sidebar.selectbox(
        "Select a template",
        [
            "Open-ended question",
            "Story continuation",
            "Code explanation",
            "Custom"
        ]
    )
    
    prompt_templates = {
        "Open-ended question": "Answer the following question thoughtfully:\n\n",
        "Story continuation": "Continue the following story:\n\n",
        "Code explanation": "Explain what the following code does:\n\n",
        "Custom": ""
    }
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt",
            value=prompt_templates[template],
            height=150
        )
        
        if st.button("Generate", type="primary"):
            if prompt:
                with st.spinner("Generating response..."):
                    # Modified to expect a tuple with stats
                    completion, stats = generate_text(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                
                st.text_area("Generated Text", value=completion, height=300)
                
                # Show performance metrics for this generation
                st.info(f"""
                    **Generation Stats**:
                    - Inference time: {stats.inference_time_ms:.2f} ms
                    - Prompt tokens: {stats.prompt_tokens}
                    - Completion tokens: {stats.completion_tokens}
                    - Total tokens: {stats.total_tokens}
                """)
                
                # Add to session state history
                if "history" not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    "prompt": prompt,
                    "completion": completion,
                    "stats": stats
                })
            else:
                st.error("Please enter a prompt.")
    
    with col2:
        st.subheader("Performance Overview")
        
        summary = monitor.get_summary_stats()
        if summary:
            st.metric("Avg. Response Time", f"{summary.get('avg_inference_time_ms', 0):.2f} ms")
            st.metric("Total Tokens Used", summary.get('total_tokens_processed', 0))
            
            # Get history from current session
            if "history" in st.session_state and st.session_state.history:
                # Recent performance chart
                history_df = pd.DataFrame([
                    {"index": i, "time_ms": item["stats"].inference_time_ms}
                    for i, item in enumerate(st.session_state.history)
                ])
                
                fig = px.line(history_df, x="index", y="time_ms", 
                              title="Response Times (This Session)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inference data yet. Generate some text to see stats.")
    
    # Session History
    if "history" in st.session_state and st.session_state.history:
        st.subheader("Session History")
        
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Interaction {len(st.session_state.history) - i}"):
                st.markdown("**Prompt:**")
                st.text(item["prompt"])
                st.markdown("**Completion:**")
                st.text(item["completion"])

if __name__ == "__main__":
    main()