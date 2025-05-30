"""
Script to download and prepare datasets.
"""
import os
import argparse
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url: str, destination: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                      help="Directory to save downloaded files")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example: Download COCO validation images
    coco_url = "http://images.cocodataset.org/zips/val2017.zip"
    download_file(coco_url, output_dir / "val2017.zip")
    
    print("Download complete. Please extract the files manually.")

if __name__ == "__main__":
    main() 