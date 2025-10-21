import streamlit as st
import requests
import json
import sys
from pathlib import Path
import requests  # type: ignore

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Audio Bot", page_icon="🤖", layout="wide")

st.title(" Audio Bot")
st.markdown("Simple command-based interface for searching audio samples")

API_URL = "http://localhost:8001"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(
        {
            "role": "bot",
            "message": " Hi! I'm Audio Bot. Type 'help' to see available commands.",
        }
    )


def parse_command(user_input):
    """Parse user command and return response"""
    command = user_input.lower().strip()

    if command == "help":
        return {
            "type": "help",
            "message": """
**Available Commands:**

**Search Commands:**
- `buscar [genre]` - Search samples by genre (e.g., "buscar techno")
- `cluster [number]` - Show samples in cluster (e.g., "cluster 0")
- `stats` - Show dataset statistics
- `anomalias` - Show anomalous samples

**Info Commands:**
- `generos` - List all available genres
- `clusters` - Show cluster information
- `help` - Show this help message

**Examples:**
- `buscar house`
- `cluster 2`
- `stats`
- `anomalias`
            """,
        }

    elif command == "stats":
        try:
            response = requests.get(f"{API_URL}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {"type": "stats", "data": data}
        except:
            return {"type": "error", "message": "Cannot connect to API"}

    elif command.startswith("buscar "):
        genre = command.replace("buscar ", "").strip()
        try:
            response = requests.get(
                f"{API_URL}/search-by-filters",
                params={"genre": genre, "limit": 10},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return {"type": "search_results", "data": data, "query": genre}
        except:
            return {"type": "error", "message": "Cannot connect to API"}

    elif command.startswith("cluster "):
        try:
            cluster_id = int(command.replace("cluster ", "").strip())
            response = requests.get(
                f"{API_URL}/search-by-filters",
                params={"cluster": cluster_id, "limit": 10},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "type": "cluster_results",
                    "data": data,
                    "cluster_id": cluster_id,
                }
        except ValueError:
            return {"type": "error", "message": "Invalid cluster number"}
        except:
            return {"type": "error", "message": "Cannot connect to API"}

    elif command == "anomalias":
        return {
            "type": "info",
            "message": "To see anomalies, run the evaluation script or check the Overview page.",
        }

    elif command == "generos":
        return {
            "type": "info",
            "message": "Available genres: techno, house, dubstep, ambient, drum_and_bass, trance",
        }

    elif command == "clusters":
        try:
            response = requests.get(f"{API_URL}/clusters", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {"type": "clusters_info", "data": data}
        except:
            return {"type": "error", "message": "Cannot connect to API"}

    else:
        return {
            "type": "unknown",
            "message": f"Unknown command: '{command}'. Type 'help' to see available commands.",
        }


def render_response(response):
    """Render bot response based on type"""
    if response["type"] == "help":
        st.markdown(response["message"])

    elif response["type"] == "stats":
        data = response["data"]
        st.markdown(f"**Total Samples:** {data['total_samples']}")
        st.markdown(f"**Embedding Dimension:** {data['embedding_dimension']}")
        st.markdown("**Genre Distribution:**")
        for genre, count in data["genres"].items():
            st.markdown(f"  • {genre}: {count}")

    elif response["type"] == "search_results":
        data = response["data"]
        st.markdown(
            f"Found **{data['total_results']}** samples for genre: **{response['query']}**"
        )
        if data["results"]:
            st.markdown("**Top results:**")
            for i, result in enumerate(data["results"][:5], 1):
                st.markdown(
                    f"{i}. `{result['filename']}` - Cluster {result['cluster']}"
                )
        else:
            st.markdown("No samples found.")

    elif response["type"] == "cluster_results":
        data = response["data"]
        st.markdown(
            f"Found **{data['total_results']}** samples in cluster **{response['cluster_id']}**"
        )
        if data["results"]:
            st.markdown("**Samples:**")
            for i, result in enumerate(data["results"][:5], 1):
                st.markdown(f"{i}. `{result['filename']}` - {result['genre']}")
        else:
            st.markdown("No samples found.")

    elif response["type"] == "clusters_info":
        data = response["data"]
        st.markdown(f"**Total Clusters:** {data['total_clusters']}")
        for cluster_id, info in list(data["clusters"].items())[:3]:
            st.markdown(f"**Cluster {cluster_id}:** {info['size']} samples")
            st.json(info["genres"])

    elif response["type"] == "info":
        st.info(response["message"])

    elif response["type"] == "error":
        st.error(response["message"])

    elif response["type"] == "unknown":
        st.warning(response["message"])


st.markdown("---")

chat_container = st.container()

with chat_container:
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(chat["message"])
        else:
            with st.chat_message("assistant"):
                if "response" in chat:
                    render_response(chat["response"])
                else:
                    st.markdown(chat["message"])

st.markdown("---")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Type a command:",
        key="user_input",
        placeholder="Type 'help' for available commands",
        label_visibility="collapsed",
    )

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

if send_button and user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    response = parse_command(user_input)

    st.session_state.chat_history.append({"role": "bot", "response": response})

    st.rerun()

st.markdown("---")

with st.expander("💡 Quick Commands"):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Stats", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "message": "stats"})
            response = parse_command("stats")
            st.session_state.chat_history.append({"role": "bot", "response": response})
            st.rerun()

    with col2:
        if st.button("Buscar Techno", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "message": "buscar techno"}
            )
            response = parse_command("buscar techno")
            st.session_state.chat_history.append({"role": "bot", "response": response})
            st.rerun()

    with col3:
        if st.button("Cluster 0", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "message": "cluster 0"}
            )
            response = parse_command("cluster 0")
            st.session_state.chat_history.append({"role": "bot", "response": response})
            st.rerun()

if st.button("🗑️ Clear Chat", use_container_width=False):
    st.session_state.chat_history = [
        {
            "role": "bot",
            "message": "Chat cleared. Type 'help' to see available commands.",
        }
    ]
    st.rerun()
