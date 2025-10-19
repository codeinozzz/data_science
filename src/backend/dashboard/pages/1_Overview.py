import streamlit as st
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")

st.title("📊 Dataset Overview")

try:
    metadata_file = PROCESSED_DIR / "metadata_with_clusters.json"
    clusters_file = PROCESSED_DIR / "clusters_analysis.json"
    k_analysis_file = PROCESSED_DIR / "k_analysis.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    with open(clusters_file, 'r') as f:
        clusters = json.load(f)
    
    with open(k_analysis_file, 'r') as f:
        k_analysis = json.load(f)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(metadata))
    
    with col2:
        unique_genres = len(set(m['genre'] for m in metadata))
        st.metric("Genres", unique_genres)
    
    with col3:
        st.metric("Clusters", len(clusters))
    
    with col4:
        best_k_idx = np.argmax(k_analysis['silhouette_scores'])
        best_silhouette = k_analysis['silhouette_scores'][best_k_idx]
        st.metric("Best Silhouette", f"{best_silhouette:.3f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Genre Distribution")
        genre_counts = {}
        for m in metadata:
            genre = m['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        df_genres = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
        df_genres = df_genres.sort_values('Count', ascending=False)
        
        st.bar_chart(df_genres.set_index('Genre'))
        
        st.dataframe(df_genres, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Sizes")
        cluster_sizes = {int(k): v['size'] for k, v in clusters.items()}
        
        df_clusters = pd.DataFrame(list(cluster_sizes.items()), columns=['Cluster', 'Samples'])
        df_clusters = df_clusters.sort_values('Cluster')
        
        st.bar_chart(df_clusters.set_index('Cluster'))
        
        st.dataframe(df_clusters, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("K-Value Analysis")
    st.markdown("Silhouette scores for different K values:")
    
    df_k = pd.DataFrame({
        'K': k_analysis['k_values'],
        'Silhouette Score': k_analysis['silhouette_scores']
    })
    
    st.line_chart(df_k.set_index('K'))
    
    st.dataframe(df_k, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Cluster Composition")
    
    for cluster_id, info in sorted(clusters.items(), key=lambda x: int(x[0])):
        with st.expander(f"Cluster {cluster_id} - {info['size']} samples"):
            st.write(f"**Genre distribution:**")
            st.json(info['genres'])
            
            st.write(f"**Sample examples:**")
            for sample in info['samples'][:5]:
                st.text(f"  • {sample}")

except FileNotFoundError as e:
    st.error("Data files not found. Please run `python scripts/analyze_clusters.py` first.")
    st.info(f"Missing file: {e.filename}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)