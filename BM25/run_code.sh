#Full dataset: 'dl19' 'dl20' 'covid' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'dbpedia' 'robust04''
#Dataset variable my be changed to reflect what datasets are needed

datasets=('dl19' 'dl20' 'covid' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'dbpedia' 'robust04')

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python bm25.py --dataset "$dataset" --k 100
done
echo "All datasets processed!"
