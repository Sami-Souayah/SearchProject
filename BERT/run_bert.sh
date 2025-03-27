export HF_HOME=/tmp/huggingface

#Full dataset: 'dl19' 'dl20' 'covid' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'dbpedia' 'robust04''
# Datasets variable under may be modified to reflect datasets already processed. 

datasets=('dl20' 'covid' 'arguana' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'quora' 'dbpedia' 'fever' 'robust04' 'signal')

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    python rerank.py --dataset "$dataset"
done

echo "All datasets processed!"