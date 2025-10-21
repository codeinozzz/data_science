import os
import time
import requests
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import requests  # type: ignore


load_dotenv()


class FreesoundDownloader:
    def __init__(self, api_key: str, output_dir: str):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.base_url = "https://freesound.org/apiv2"
        self.headers = {"Authorization": f"Token {api_key}"}

    def search_sounds(self, query: str, max_results: int = 50) -> List[Dict]:
        sounds: list[dict] = []
        page = 1

        while len(sounds) < max_results:
            params = {
                "query": query,
                "filter": "duration:[0 TO 5]",
                "fields": "id,name,previews,duration,tags",
                "page_size": min(150, max_results - len(sounds)),
                "page": page,
            }

            try:
                response = requests.get(
                    f"{self.base_url}/search/text/",
                    headers=self.headers,
                    params=params,  # type: ignore
                    timeout=10,
                )

                if response.status_code != 200:
                    print(f"Error in search: {response.status_code}")
                    break

                data = response.json()
                sounds.extend(data.get("results", []))

                if not data.get("next"):
                    break

                page += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"Exception during search: {e}")
                break

        return sounds[:max_results]

    def download_sound(self, sound_id: int, output_path: Path) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/sounds/{sound_id}/", headers=self.headers, timeout=10
            )

            if response.status_code != 200:
                return False

            sound_data = response.json()
            preview_url = sound_data["previews"]["preview-hq-mp3"]

            audio_response = requests.get(preview_url, timeout=30)

            if audio_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(audio_response.content)
                return True

        except Exception as e:
            print(f"Error downloading {sound_id}: {e}")

        return False

    def download_dataset(
        self, queries_by_genre: Dict[str, List[str]], samples_per_query: int = 10
    ):
        total_downloaded = 0

        for genre, queries in queries_by_genre.items():
            print(f"\nProcessing genre: {genre}")
            genre_dir = self.output_dir / genre
            genre_dir.mkdir(parents=True, exist_ok=True)

            for query in queries:
                print(f"  Searching: {query}")
                sounds = self.search_sounds(query, samples_per_query)
                print(f"  Found: {len(sounds)} sounds")

                for idx, sound in enumerate(sounds, 1):
                    sound_id = sound["id"]
                    filename = f"{genre}_{sound_id}.mp3"
                    output_path = genre_dir / filename

                    if output_path.exists():
                        print(f"    [{idx}/{len(sounds)}] Skip: {filename}")
                        continue

                    if self.download_sound(sound_id, output_path):
                        total_downloaded += 1
                        print(f"    [{idx}/{len(sounds)}] Downloaded: {filename}")
                    else:
                        print(f"    [{idx}/{len(sounds)}] Failed: {filename}")

                    time.sleep(1)

        return total_downloaded


def main():
    API_KEY = os.getenv("FREESOUND_API_KEY")

    if not API_KEY:
        print("ERROR: FREESOUND_API_KEY not found")
        return

    print(f"API Key loaded: {API_KEY[:10]}...")

    OUTPUT_DIR = "data/raw"

    queries = {
        "techno": [
            "techno kick",
            "techno hihat",
            "techno clap",
            "techno bass loop",
            "techno synth",
        ],
        "house": [
            "house kick",
            "house clap",
            "house shaker",
            "house bass",
            "house vocal",
        ],
        "dubstep": [
            "dubstep wobble",
            "dubstep snare",
            "dubstep kick",
            "dubstep bass",
            "dubstep riser",
        ],
        "ambient": [
            "ambient pad",
            "ambient texture",
            "ambient drone",
            "ambient chime",
            "ambient atmosphere",
        ],
        "drum_and_bass": [
            "dnb kick",
            "dnb snare",
            "dnb bass",
            "dnb break",
            "dnb reese",
        ],
        "trance": [
            "trance kick",
            "trance pluck",
            "trance lead",
            "trance pad",
            "trance riser",
        ],
        "jazz": [
            "jazz piano",
            "jazz saxophone",
            "jazz trumpet",
            "jazz bass",
            "jazz drums",
            "jazz guitar",
            "jazz cymbal",
            "jazz vocal",
        ],
        "rock": [
            "rock guitar",
            "rock drum",
            "rock bass",
            "rock vocal",
            "rock snare",
            "rock kick",
            "rock cymbal",
            "rock riff",
        ],
        "classical": [
            "classical piano",
            "classical violin",
            "classical cello",
            "classical flute",
            "classical orchestra",
            "classical strings",
            "classical brass",
            "classical timpani",
        ],
        "experimental": [
            "noise",
            "glitch",
            "field recording",
            "abstract sound",
            "industrial sound",
            "atonal",
            "avant garde",
            "sound design",
        ],
        "world": [
            "ethnic percussion",
            "didgeridoo",
            "sitar",
            "tabla",
            "african drum",
            "shakuhachi",
            "gamelan",
            "throat singing",
        ],
        "hip_hop": [
            "hip hop beat",
            "rap vocal",
            "808 bass",
            "trap snare",
            "hip hop kick",
            "vinyl scratch",
            "boom bap",
            "hip hop hi hat",
        ],
    }

    print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
    print("Starting download...\n")

    downloader = FreesoundDownloader(API_KEY, OUTPUT_DIR)
    total = downloader.download_dataset(queries, samples_per_query=5)

    print(f"\n{'='*50}")
    print(f"Download complete: {total} samples downloaded")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
