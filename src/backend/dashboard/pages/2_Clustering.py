import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR

st.set_page_config(page_title="Clustering Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Clustering Explorer")
st.markdown("Interactive 2D visualization of audio samples")

try:
    embeddings_2d = np.load(PROCESSED_DIR / "embeddings_2d_tsne.npy")
    labels = np.load(PROCESSED_DIR / "cluster_labels.npy")
    
    with open(PROCESSED_DIR / "metadata_with_clusters.json", 'r') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': labels,
        'genre': [m['genre'] for m in metadata],
        'filename': [m['filename'] for m in metadata],
        'duration': [m['duration'] for m in metadata]
    })
    
    st.sidebar.header("Visualization Options")
    
    color_by = st.sidebar.radio(
        "Color by:",
        ["Cluster", "Genre"],
        index=0
    )
    
    show_labels = st.sidebar.checkbox("Show point labels", value=False)
    
    filter_genre = st.sidebar.multiselect(
        "Filter by genre:",
        options=sorted(df['genre'].unique()),
        default=None
    )
    
    filter_cluster = st.sidebar.multiselect(
        "Filter by cluster:",
        options=sorted(df['cluster'].unique()),
        default=None
    )
    
    df_filtered = df.copy()
    
    if filter_genre:
        df_filtered = df_filtered[df_filtered['genre'].isin(filter_genre)]
    
    if filter_cluster:
        df_filtered = df_filtered[df_filtered['cluster'].isin(filter_cluster)]
    
    st.sidebar.markdown(f"**Showing:** {len(df_filtered)} / {len(df)} samples")
    
    color_column = 'cluster' if color_by == "Cluster" else 'genre'
    
    fig = px.scatter(
        df_filtered,
        x='x',
        y='y',
        color=color_column,
        hover_data=['filename', 'genre', 'cluster', 'duration'],
        title=f"Audio Samples - Colored by {color_by}",
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(showgrid=True, gridcolor='white'),
        yaxis=dict(showgrid=True, gridcolor='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Details")
        
        if st.session_state.get('selected_sample'):
            sample = st.session_state.selected_sample
            st.json(sample)
        else:
            st.info("Click on a point in the plot to see details")
    
    with col2:
        st.subheader("Filtered Samples")
        st.dataframe(
            df_filtered[['filename', 'genre', 'cluster', 'duration']].head(20),
            use_container_width=True
        )
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips
    - **Hover** over points to see sample info
    - **Zoom** by drawing a box
    - **Pan** by clicking and dragging
    - **Reset** by double-clicking
    - Use **filters** in the sidebar to explore specific genres or clusters
    """)

except FileNotFoundError:
    st.error("Clustering data not found. Please run `python scripts/analyze_clusters.py` first.")
except Exception as e:
    st.error(f"Error loading visualization: {e}")
    st.exception(e)