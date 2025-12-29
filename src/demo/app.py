import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import sys
import torch
import time
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.validator import Validator
from src.core.search_space import Genome
from src.core.proxy_evaluator import ProxyEvaluator

# Config
CHECKPOINT_PATH = "./data/logs/search_tinyllama_wandb/checkpoint_gen_50.json"
ARTIFACTS_DIR = "./deployment/artifacts"
PROFILE_PATH = "./data/profiles/tinyllama_sensitivity.json"

st.set_page_config(page_title="EMPAS Dashboard", layout="wide")

@st.cache_data
def load_search_data():
    """Parses the checkpoint to visualize the search history."""
    if not os.path.exists(CHECKPOINT_PATH):
        st.warning(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return None
    
    with open(CHECKPOINT_PATH, 'r') as f:
        data = json.load(f)
    
    evaluator = ProxyEvaluator(PROFILE_PATH)
    
    records = []
    for i, genes in enumerate(data['population']):
        genome = Genome(genes=genes)
        fit = evaluator.evaluate(genome)
        records.append({
            "id": i,
            "loss": fit.validation_loss,
            "vram": fit.vram_peak_mb,
            "latency": fit.latency_ms,
            "genes": str(genes)
        })
    return pd.DataFrame(records)

@st.cache_data
def load_artifacts():
    """Loads the 3 exported archetypes."""
    artifacts = {}
    for name in ["max_accuracy", "balanced", "max_compression"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                artifacts[name] = json.load(f)
    return artifacts

# Global cache for the model wrapper to avoid repeated downloads
class ModelContainer:
    def __init__(self):
        self.validator = None
        self.current_config_name = None

@st.cache_resource
def get_model_container():
    return ModelContainer()

def load_active_model(config_name, config_genes):
    """
    Loads or reloads the model with the specific quantization map.
    """
    container = get_model_container()
    
    if container.current_config_name != config_name:
        
        with st.spinner(f"Switching Architecture to [{config_name}]... (Reloading Weights)"):
            # 1. Clear GPU memory if possible
            if container.validator:
                del container.validator
                gc.collect()
                torch.cuda.empty_cache()
            
            # 2. Init new validator (Loads Base FP16 Model)
            # Using TinyLlama
            container.validator = Validator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            # 3. Apply Genome
            genome = Genome(genes=config_genes)
            container.validator.apply_genome(genome)
            
            container.current_config_name = config_name
            
    return container.validator

# --- GUI LAYOUT ---

st.title("🧬 EMPAS: Evolutionary Mixed-Precision Architecture Search")
st.markdown("### Interactive Portfolio Demo")

# Sidebar: Controls
st.sidebar.header("Model Configuration")

artifacts = load_artifacts()
search_df = load_search_data()

# Selection
selected_archetype = st.sidebar.radio(
    "Select Architecture",
    ["balanced", "max_accuracy", "max_compression", "Baseline (FP16)"]
)

# Logic to get genes for selection
if selected_archetype == "Baseline (FP16)":
    # 22 layers of 16-bit
    current_genes = [16] * 22
    metrics = {"predicted_loss": 0.88, "predicted_vram_mb": 2200}
else:
    if selected_archetype in artifacts:
        data = artifacts[selected_archetype]
        # Extract genes from the artifact map
        qmap = data['config']['quantization_map']
        sorted_keys = sorted(qmap.keys(), key=lambda x: int(x.split('_')[1]))
        current_genes = [qmap[k] for k in sorted_keys]
        metrics = data['metrics']
    else:
        st.error("Artifact not found. Run export_artifacts.py first.")
        st.stop()

# Display Metrics
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Predicted Loss", f"{metrics['predicted_loss']:.4f}")
col2.metric("VRAM (MB)", f"{metrics.get('predicted_vram_mb', 0):.0f}")
col3.metric("Compression", f"{(1 - (sum(current_genes)/(16*22)))*100:.1f}%")

# --- TAB 1: VISUALIZATION ---
tab1, tab2 = st.tabs(["📊 Search Analysis", "💬 Inference Playground"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Pareto Frontier (Loss vs VRAM)")
        if search_df is not None:
            # Highlight selected point
            # Approximate matching based on loss/vram
            
            fig = px.scatter(
                search_df, 
                x="vram", 
                y="loss", 
                hover_data=["genes"],
                color="loss",
                title="Population Distribution (Generation 50)"
            )
            # Add current selection as a big red dot
            fig.add_scatter(
                x=[metrics.get('predicted_vram_mb', 0)],
                y=[metrics['predicted_loss']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='Selected'
            )
            st.plotly_chart(fig, width='stretch')
    
    with col_b:
        st.subheader("Genome Inspector (Layer Bit-Widths)")
        
        # Create a dataframe for the bar chart
        layer_df = pd.DataFrame({
            "Layer": [f"L{i}" for i in range(len(current_genes))],
            "Bit-Width": current_genes
        })
        
        # Color mapping logic manually
        # 2->Red, 4->Orange, 8->Yellow, 16->Green
        
        fig_bar = px.bar(
            layer_df, 
            x="Layer", 
            y="Bit-Width", 
            color="Bit-Width",
            title="Quantization Map",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_bar, width='stretch')
        
        st.code(str(current_genes), language="json")

# --- TAB 2: INFERENCE ---
with tab2:
    st.subheader(f"Live Inference: {selected_archetype.upper()}")
    
    # Load Model (Lazy loading only when in this tab or needed)
    if st.button("Load/Reload Model"):
        model = load_active_model(selected_archetype, current_genes)
        st.success(f"Loaded {selected_archetype} successfully!")

    prompt = st.text_area("Input Prompt", "The future of Artificial Intelligence is")
    
    if st.button("Generate"):
        # Ensure model is loaded
        container = get_model_container()
        if not container.validator or container.current_config_name != selected_archetype:
             model = load_active_model(selected_archetype, current_genes)
        else:
             model = container.validator

        with st.spinner("Generating..."):
            # Use the internal wrapper logic
            tokenizer = model.wrapper.tokenizer
            net = model.wrapper.model
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.wrapper.device)
            
            start_t = time.time()
            with torch.no_grad():
                outputs = net.generate(
                    inputs.input_ids, 
                    max_new_tokens=64, 
                    do_sample=True, 
                    temperature=0.7
                )
            end_t = time.time()
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.markdown("### Output")
            st.write(output_text)
            st.caption(f"Latency: {(end_t - start_t)*1000:.2f} ms")