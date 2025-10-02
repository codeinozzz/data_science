import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.storage.chroma_client import ChromaStorage

db_path = Path(__file__).parent.parent / "chroma_db"
storage = ChromaStorage(persist_directory=str(db_path))

print(f"Count: {storage.count()}")

all_data = storage.get_all_samples()
print(f"\nType: {type(all_data)}")
print(f"Keys: {all_data.keys()}")

for key in all_data.keys():
    value = all_data[key]
    print(f"\n{key}:")
    print(f"  Type: {type(value)}")
    if value:
        print(f"  Length: {len(value)}")
        if isinstance(value, list) and len(value) > 0:
            print(f"  First item type: {type(value[0])}")
