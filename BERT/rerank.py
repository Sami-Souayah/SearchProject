import pyserini
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from indexpaths import THE_INDEX,THE_TOPICS
import argparse
import json
import csv
from tqdm import tqdm
import pandas as pd
import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory = '/home/gridsan/ssouayah/BM25Output'
OutputDir = '/home/gridsan/ssouayah/BERTOutput'
chunk_size = 500
model_name = '/home/gridsan/ssouayah/ms-marco-MiniLM-L6-v2'
# This is minimal code to get a reranker running. You'll need to fill in the blanks, but this is
# the structure you need for the code.


# We need a function that takes in the docid from the bm25 retrival run and outputs
# the document text. 
def load_text(dataset, searcher, docid): 
    doc = searcher.doc(docid)
    json_doc = json.loads(doc.raw())
    if dataset == 'dl19' or dataset == 'dl20':
        text = json_doc['contents']
    else:
        if 'title' in json_doc:
            text = f"{json_doc['title']} {json_doc['text']}"
    return text 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Input dataset', type=str)


    args = parser.parse_args()
    dataset = args.dataset
    bm25_run = pd.read_csv(f'/home/gridsan/ssouayah/BM25Output/{dataset}_run.csv', delimiter=' ', dtype=str)
    bm25_run.columns = ['qid', 'q0', 'docid', 'rank', 'score','extra','extra']
    searcher =  LuceneSearcher.from_prebuilt_index(THE_INDEX[dataset])
    topics = get_topics(THE_TOPICS[dataset] if dataset != 'dl20' else 'dl20')
    topics = {str(key): value for key, value in topics.items()}    
    qrels = get_qrels(THE_TOPICS[dataset])
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reranked_run = [] 
    for qid in tqdm(topics, desc="Processing Queries"):
        qid = str(qid)
        # Fetch BM25 results for the given topic
        query = topics[qid]['title']
        bm25_query_results = bm25_run[bm25_run['qid'] == qid]
        # This should be 1000 x num_columns....
        query_reranked = [] 
        for id, row in tqdm(bm25_query_results.iterrows(), desc=f"Reranking Docs for Query {qid}", leave=False):
            docid = row['docid']
            document_text = load_text(dataset, searcher, docid)

            encoding = tokenizer(
                            query, document_text, 
                            padding=True, 
                            truncation=True, 
                            return_tensors="pt", 
                    ).to(device)
            
            # We need the score of only the positive label, usually at index 1.
            # 0 = Prob of non-relevant
            # 1 = Prob of relevant
            with torch.no_grad():
                prob_relevant = model(**encoding).logits.item()
            
            query_reranked.append([qid, docid, prob_relevant])

        sorted_query_reranked = sorted(query_reranked,key=lambda x: x[2], reverse=True)
        reranked_run.append(sorted_query_reranked)



    with open(f'/home/gridsan/ssouayah/BERTOutput/{dataset}_BERT.csv', 'w', newline='') as file:
        for query_results in reranked_run:
            rank = 0
            for document in query_results:
                file.write(f'{document[0]} Q0 {document[1]} {rank} {document[2]} rank \n')
                rank += 1

print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[dataset]} '/home/gridsan/ssouayah/BERTOutput/{dataset}_BERT.csv'"))



