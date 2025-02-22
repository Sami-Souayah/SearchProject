for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python3 bm25.py --dataset "$dataset" --k 1000
done
echo "All datasets processed!"
