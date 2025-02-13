datasets=('dl19' 'covid' 'arguana' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'quora' 'dbpedia' 'fever' 'robust04' 'signal')

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python3 bm25.py --dataset "$dataset" --k 100
done

echo "All datasets processed!