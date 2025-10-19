from pathlib import Path
from fastapi import HTTPException
import re


def validate_audio_file(filename, file_size, max_size_mb=10):
    allowed = ['.mp3', '.wav', '.flac']
    ext = Path(filename).suffix.lower()
    
    if ext not in allowed:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed}")
    
    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(413, f"File too large. Max: {max_size_mb}MB")
    
    safe_filename = re.sub(r'[^\w\s.-]', '', filename)
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(400, "Invalid filename characters")
    
    return True


def validate_search_params(n_results):
    if n_results < 1 or n_results > 20:
        raise HTTPException(400, "n_results must be between 1 and 20")
    return True