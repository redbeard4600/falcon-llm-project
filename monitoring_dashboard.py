# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monitoring import monitor

def load_monitoring_data():
    """Convert monitoring data to pandas DataFrame"""
    if not monitor.inference_history:
        return pd.DataFrame()
    
    data = []
    for stat in monitor.inference_history:
        data.append({
            "timestamp": stat.timestamp,
            "prompt_tokens": stat.prompt_tokens,
            "completion_tokens": stat.completion_tokens,
            "total_tokens": stat.total_tokens,
            "inference_time_ms": stat.inference_time_ms,
            "memory_usage_mb": stat.memory_usage_mb
        })
    
    return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="Falcon 7B Monitor", page_icon="📊", layout="wide")
    
    st.title("Falcon 7B Model Monitoring Dashboard")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    
    if auto_refresh:
        st.sidebar.text(f"Auto-refreshing every {refresh_interval} seconds")
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    summary = monitor.get_summary_stats()
    
    with col1:
        st.metric(label="Total Inferences", value=summary.get("total_inferences", 0))
    
    with col2:
        st.metric(label="Avg. Inference Time (ms)", 
                  value=round(summary.get("avg_inference_time_ms", 0), 2))
    
    with col3:
        st.metric(label="Total Tokens Processed", 
                  value=summary.get("total_tokens_processed", 0))
    
    # Load data
    df = load_monitoring_data()
    
    if not df.empty:
        # Time Series Charts
        st.subheader("Inference Performance Over Time")
        tab1, tab2, tab3 = st.tabs(["Inference Time", "Token Usage", "Memory Usage"])
        
        with tab1:
            fig = px.line(df, x="timestamp", y="inference_time_ms", 
                          title="Inference Time (ms)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.area(df, x="timestamp", y=["prompt_tokens", "completion_tokens"], 
                          title="Token Usage")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.line(df, x="timestamp", y="memory_usage_mb", 
                          title="Memory Usage (MB)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("Performance Relationships")
        fig = px.scatter(df, x="total_tokens", y="inference_time_ms", 
                         size="memory_usage_mb", hover_data=["timestamp"],
                         title="Relationship between Tokens and Inference Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw Data View
        st.subheader("Recent Inference Records")
        st.dataframe(df.sort_values("timestamp", ascending=False).head(10))
    else:
        st.info("No inference data available yet. Run some inferences to populate the dashboard.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()