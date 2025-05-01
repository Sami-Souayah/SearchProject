export HF_HOME=/tmp/huggingface

#Full dataset: 'dl19' 'dl20' 'covid' 'touche' 'news' 'scifact' 'fiqa' 'scidocs' 'nfc' 'dbpedia' 'robust04''
#Full models: '/home/gridsan/ssouayah/ms-marco-MiniLM-L6-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L4-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L2-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L12-v2'
# Datasets variable under may be modified to reflect datasets already processed. 


datasets=('signal')
models=('/home/gridsan/ssouayah/ms-marco-MiniLM-L6-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L4-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L2-v2' '/home/gridsan/ssouayah/ms-marco-MiniLM-L12-v2')

for model in "${models[@]}"; do
    echo "Processing $model..."
    for dataset in "${datasets[@]}"; do
        echo "Processing $dataset..."
        python sntnctransformers.py --dataset "$dataset" --model "$model"
    done
done

echo "All datasets processed!"