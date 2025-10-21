import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DIR, MODELS_DIR
from src.models.clustering import AudioClusterer

st.set_page_config(page_title="Clustering Explorer", layout="wide")

st.title("Clustering Explorer")
st.markdown("Interactive 2D visualization with clusters, centroids, and anomalies")

try:
    embeddings_2d = np.load(PROCESSED_DIR / "embeddings_2d_tsne.npy")
    labels = np.load(PROCESSED_DIR / "cluster_labels.npy")

    with open(PROCESSED_DIR / "metadata_with_anomalies.json", "r") as f:
        metadata = json.load(f)

    centroids_2d = None
    has_centroids = False
    
    try:
        centroids_2d = np.load(PROCESSED_DIR / "centroids_2d.npy")
        has_centroids = True
    except:
        pass

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": labels,
        "genre": [m["genre"] for m in metadata],
        "filename": [m["filename"] for m in metadata],
        "duration": [m["duration"] for m in metadata],
        "is_anomaly": [m.get("is_anomaly", False) for m in metadata],
        "anomaly_score": [m.get("anomaly_score", 0) for m in metadata]
    })

    st.sidebar.header("Visualization Options")
    color_by = st.sidebar.radio("Color by:", ["Cluster", "Genre"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Show/Hide Elements")
    
    show_centroids = st.sidebar.checkbox("Show centroids", value=has_centroids) if has_centroids else False
    show_anomalies = st.sidebar.checkbox("Highlight anomalies", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    filter_genre = st.sidebar.multiselect(
        "Filter by genre:", options=sorted(df["genre"].unique()), default=None
    )

    filter_cluster = st.sidebar.multiselect(
        "Filter by cluster:", options=sorted(df["cluster"].unique()), default=None
    )

    show_only_anomalies = st.sidebar.checkbox("Show ONLY anomalies", value=False)

    df_filtered = df.copy()

    if filter_genre:
        df_filtered = df_filtered[df_filtered["genre"].isin(filter_genre)]

    if filter_cluster:
        df_filtered = df_filtered[df_filtered["cluster"].isin(filter_cluster)]

    if show_only_anomalies:
        df_filtered = df_filtered[df_filtered["is_anomaly"] == True]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing:** {len(df_filtered)} / {len(df)} samples")

    color_column = "cluster" if color_by == "Cluster" else "genre"

    if show_anomalies:
        df_normal = df_filtered[df_filtered["is_anomaly"] == False]
        df_anomalies = df_filtered[df_filtered["is_anomaly"] == True]
    else:
        df_normal = df_filtered
        df_anomalies = pd.DataFrame()

    fig = px.scatter(
        df_normal,
        x="x",
        y="y",
        color=color_column,
        hover_data=["filename", "genre", "cluster", "duration"],
        title=f"Audio Samples - Colored by {color_by}",
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")),
        selector=dict(mode="markers"),
    )

    if show_anomalies and len(df_anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_anomalies["x"],
                y=df_anomalies["y"],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=12,
                    color='#FF4B4B',
                    opacity=0.9,
                    line=dict(width=2, color='darkred'),
                    symbol='diamond'
                ),
                text=df_anomalies["filename"],
                customdata=df_anomalies["anomaly_score"],
                hovertemplate='<b>%{text}</b><br>Genre: ' + df_anomalies["genre"] + '<br>Cluster: ' + df_anomalies["cluster"].astype(str) + '<br>Anomaly Score: %{customdata:.4f}<extra></extra>',
                showlegend=True
            )
        )

    if has_centroids and show_centroids:
        if filter_cluster:
            centroids_to_show = centroids_2d[filter_cluster]
            cluster_ids = filter_cluster
        else:
            centroids_to_show = centroids_2d
            cluster_ids = list(range(len(centroids_2d)))

        fig.add_trace(
            go.Scatter(
                x=centroids_to_show[:, 0],
                y=centroids_to_show[:, 1],
                mode='markers+text',
                name='Centroids',
                marker=dict(
                    symbol='x',
                    size=25,
                    color='black',
                    line=dict(width=4, color='yellow')
                ),
                text=[f'C{i}' for i in cluster_ids],
                textposition='top center',
                textfont=dict(size=14, color='black', family='Arial Black'),
                hovertemplate='<b>Centroid %{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                showlegend=True
            )
        )

    fig.update_layout(
        height=650,
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.5)",
        xaxis=dict(showgrid=True, gridcolor="white"),
        yaxis=dict(showgrid=True, gridcolor="white"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df_filtered))
    
    with col2:
        st.metric("Clusters", df_filtered["cluster"].nunique())
    
    with col3:
        st.metric("Anomalies", df_filtered["is_anomaly"].sum())
    
    with col4:
        pct = (df_filtered["is_anomaly"].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("Anomaly Rate", f"{pct:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Centroids")
        
        if has_centroids:
            centroid_info = []
            for cluster_id in range(len(centroids_2d)):
                mask = labels == cluster_id
                n_samples = np.sum(mask)
                anomalies_in_cluster = sum(m.get("is_anomaly", False) and m.get("cluster") == cluster_id for m in metadata)
                genres_in_cluster = [m["genre"] for m, in_cluster in zip(metadata, mask) if in_cluster]
                most_common_genre = max(set(genres_in_cluster), key=genres_in_cluster.count) if genres_in_cluster else "N/A"
                
                centroid_info.append({
                    "Cluster": f"C{cluster_id}",
                    "Samples": n_samples,
                    "Anomalies": anomalies_in_cluster,
                    "Main Genre": most_common_genre
                })
            
            st.dataframe(pd.DataFrame(centroid_info), use_container_width=True, hide_index=True)
        else:
            st.info("Run KMeans clustering to see centroids")

    with col2:
        st.subheader("Top Anomalies")
        df_anom = df_filtered[df_filtered["is_anomaly"]].sort_values("anomaly_score")
        
        if len(df_anom) > 0:
            st.dataframe(df_anom[["filename", "genre", "cluster", "anomaly_score"]].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("No anomalies in filtered data")

except Exception as e:
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())