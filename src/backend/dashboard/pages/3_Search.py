import streamlit as st
import requests
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Search Samples", page_icon="🎯", layout="wide")

st.title("Search & Upload")
st.markdown("Find similar samples or add new ones to the database")

API_URL = "http://localhost:8001"

tab1, tab2, tab3 = st.tabs(["🔍 Semantic Search", "📤 Upload Sample", "🔎 Filter Search"])

with tab1:
    st.header("Semantic Search")
    st.markdown("Upload an audio file to find similar samples")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'flac'],
        help="Upload MP3, WAV, or FLAC files (max 10MB)"
    )
    
    n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if st.button("Search Similar Samples", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload an audio file first")
        else:
            with st.spinner("Searching..."):
                try:
                    files = {'file': uploaded_file}
                    params = {'n_results': n_results}
                    
                    response = requests.post(
                        f"{API_URL}/search",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(f"Found {len(data['results'])} similar samples!")
                        
                        st.subheader("Query Info")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Filename", data['query']['filename'])
                        with col2:
                            st.metric("Duration", f"{data['query']['duration']:.2f}s")
                        
                        st.subheader("Similar Samples")
                        
                        for i, result in enumerate(data['results'], 1):
                            similarity = (1 - result['distance']) * 100
                            
                            with st.expander(f"#{i} - {result['filename']} ({similarity:.1f}% similar)"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Genre:** {result['genre']}")
                                with col2:
                                    st.write(f"**Distance:** {result['distance']:.4f}")
                                with col3:
                                    st.write(f"**Similarity:** {similarity:.1f}%")
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.json(response.json())
                
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the API is running on http://localhost:8001")
                    st.info("Run: `python src/api/main.py`")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("Upload New Sample")
    st.markdown("Add a new audio sample to the database")
    
    upload_file = st.file_uploader(
        "Choose an audio file to upload",
        type=['mp3', 'wav', 'flac'],
        key="upload_file"
    )
    
    genre = st.selectbox(
        "Select genre",
        ["techno", "house", "dubstep", "ambient", "drum_and_bass", "trance"]
    )
    
    if st.button("Upload to Database", type="primary"):
        if upload_file is None:
            st.warning("Please select a file first")
        else:
            with st.spinner("Processing and uploading..."):
                try:
                    files = {'file': upload_file}
                    params = {'genre': genre}
                    
                    response = requests.post(
                        f"{API_URL}/ingest",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(data['message'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Sample ID", data['sample_id'])
                        with col2:
                            st.metric("Assigned Cluster", data['cluster'])
                        with col3:
                            st.metric("Total Samples", data['total_samples'])
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.json(response.json())
                
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Filter Search")
    st.markdown("Search samples by genre or cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filter_genre = st.selectbox(
            "Filter by genre (optional)",
            ["All"] + ["techno", "house", "dubstep", "ambient", "drum_and_bass", "trance"]
        )
    
    with col2:
        filter_cluster = st.selectbox(
            "Filter by cluster (optional)",
            ["All"] + list(range(0, 10))
        )
    
    filter_limit = st.slider("Max results", min_value=5, max_value=100, value=20)
    
    if st.button("Apply Filters", type="primary"):
        with st.spinner("Searching..."):
            try:
                params = {'limit': filter_limit}
                
                if filter_genre != "All":
                    params['genre'] = filter_genre
                
                if filter_cluster != "All":
                    params['cluster'] = filter_cluster
                
                response = requests.get(
                    f"{API_URL}/search-by-filters",
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"Found {data['total_results']} samples")
                    
                    st.subheader("Filters Applied")
                    st.json(data['filters_applied'])
                    
                    st.subheader("Results")
                    
                    for i, result in enumerate(data['results'][:20], 1):
                        with st.expander(f"#{i} - {result['filename']}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Genre:** {result['genre']}")
                            with col2:
                                st.write(f"**Cluster:** {result['cluster']}")
                            with col3:
                                st.write(f"**Duration:** {result['duration']:.2f}s")
                else:
                    st.error(f"Error: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.info(" **Tip:** Make sure the API is running before using search features")