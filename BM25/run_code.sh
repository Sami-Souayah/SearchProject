#Original dataset: 'dl19' 'dl20' 'covid' 'arguana' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'quora' 'dbpedia' 'fever' 'robust04' 'signal'
#Dataset variable my be changed to reflect what datasets are needed

datasets=('robust04' 'signal')

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python bm25.py --dataset "$dataset" --k 1000
done
echo "All datasets processed!"
