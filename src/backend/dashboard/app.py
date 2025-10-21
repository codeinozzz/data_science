import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Audio Samples Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(" Audio Samples Explorer")
st.markdown("### Semantic Search & Clustering Dashboard")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**Dataset Overview**")
    st.markdown("View statistics and metrics")
    if st.button("Go to Overview", use_container_width=True):
        st.switch_page("pages/1_Overview.py")

with col2:
    st.success("**Clustering Explorer**")
    st.markdown("Interactive 2D visualization")
    if st.button("Go to Clustering", use_container_width=True):
        st.switch_page("pages/2_Clustering.py")

with col3:
    st.warning(" **Search & Upload**")
    st.markdown("Find similar samples")
    if st.button("Go to Search", use_container_width=True):
        st.switch_page("pages/3_Search.py")

with col4:
    st.error("**Audio Bot**")
    st.markdown("Command-based search")
    if st.button("Go to Chatbot", use_container_width=True):
        st.switch_page("pages/4_Chatbot.py")

st.markdown("---")

st.markdown(
    """
### Features

- **Semantic Search**: Upload audio and find similar samples
- **Clustering Visualization**: Interactive 2D scatter plots
- **Filter by Genre/Cluster**: Advanced filtering
- **Sample Upload**: Add new samples to database
- **Chatbot Interface**: Simple command-based search

### Current Dataset

Navigate to **Overview** to see detailed statistics.
"""
)

st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
- **Overview**: Dataset statistics
- **Clustering**: Visual exploration
- **Search**: Find similar samples
- **Chatbot**: Command interface
"""
)

st.sidebar.markdown("---")
st.sidebar.info("**Version:** 1.0.0")
