#!/bin/bash

# Activate virtual environment if using one
# source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p data/{raw,processed,splits}
mkdir -p results/reproducibility

# Download and preprocess data
echo "Downloading data..."
python scripts/download_data.py --output-dir data/raw

echo "Preprocessing data..."
python scripts/preprocess.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --size 256

# Run the experiment
echo "Running CroPA experiment..."
python src/croppa/attack.py \
    --config experiments/repro_results/config.yaml \
    --output-dir results/reproducibility \
    --seed 42

echo "Experiment complete! Results saved in results/reproducibility" 