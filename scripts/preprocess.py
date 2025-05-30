"""
Script to preprocess datasets.
"""
import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

def preprocess_images(input_dir: Path, output_dir: Path, size: int = 256) -> None:
    """Preprocess images: resize, normalize, and save as tensors."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(list(input_dir.glob("*.jpg"))):
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            # Save preprocessed tensor
            output_path = output_dir / f"{img_path.stem}.pt"
            torch.save(img_tensor, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                      help="Directory containing raw images")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                      help="Directory to save processed images")
    parser.add_argument("--size", type=int, default=256,
                      help="Size to resize images to")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Preprocessing images from {input_dir} to {output_dir}")
    preprocess_images(input_dir, output_dir, args.size)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 