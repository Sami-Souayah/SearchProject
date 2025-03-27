#!/bin/bash
#SBATCH -o rerank.log-%j  # Output log file
#SBATCH -e rerank.err-%j  # Error log file
#SBATCH -t 20:00:00       # Set time limit (adjust as needed)
#SBATCH -p volta          # Specify GPU partition
#SBATCH --gres=gpu:1      # Request 1 GPU
#SBATCH -c 4              # Request 4 CPU cores
#SBATCH --mem=16G         # Request 16GB RAM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@mit.edu  # Replace with your MIT email

# Load environment
source /etc/profile
module load anaconda/2020a 

conda activate pyserini

# Set Hugging Face cache directory
export HF_HOME=/tmp/huggingface

# Define datasets
datasets=('touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'quora' 'dbpedia' 'fever' 'robust04' 'signal')

# Loop through datasets and run the Python script
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python rerank.py --dataset "$dataset"
done

echo "All datasets processed!"
