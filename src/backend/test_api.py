import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FREESOUND_API_KEY")

if api_key:
    print(f"API Key loaded successfully: {api_key[:10]}...")
else:
    print("ERROR: API Key not found")
    print(f"Current dir: {os.getcwd()}")
    print(f"Looking for .env in parent dir")
