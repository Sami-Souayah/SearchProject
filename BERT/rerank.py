import pyserini
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from indexpaths import THE_INDEX,THE_TOPICS
import argparse
import json
import csv
import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from pyserini.eval.trec_eval import evaluate


directory = '/home/gridsan/ssouayah/BM25Output'
OutputDir = '/home/gridsan/ssouayah/BERTOutput'

class FetchText():
    def __init__(self,dataset):
        self.dataset = dataset
        self.topics = get_topics(THE_TOPICS[self.dataset] if self.dataset != 'dl20' else 'dl20')
        self.qid = 0
        self.text = {}
        self.tokenized_text = {}
        self.model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
        self.directory = f"/home/gridsan/ssouayah/BM25Output/{dataset}_run.csv"
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
        self.final_scores = {}
    
    def FetchText(self, docid, qid):
        searcher =  LuceneSearcher.from_prebuilt_index(THE_INDEX[self.dataset])
        doc = searcher.doc(docid)
        json_doc = json.loads(doc.raw())
        if self.dataset == 'dl19' or self.dataset == 'dl20':
            text = json_doc['contents']
        else:
            self.text[docid] = json_doc['text']
            if 'title' in json_doc:
                text = f"{json_doc['title']} {json_doc['text']}"
        self.text[docid] = text
        self.tokenized_text[docid] = self.tokenize_text(text, qid)

    def ReadCSV(self):
        with open(self.directory, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=' ')
            for i,row in enumerate(reader):
                docID = row[2]
                self.qid = int(row[0])
                bm25score = row[4]
                self.FetchText(docID, self.qid)

    def tokenize_text(self, text, qid):
        qid = int(qid)
        query = self.topics[qid]['title']
        encoding = self.tokenizer(
            text, query,
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
        )
        return encoding
    
    def ModelEval(self):
        model = self.model
        model.eval()
        for i in self.tokenized_text:
            with torch.no_grad():
                scores = model(**self.tokenized_text[i]).logits.item()
            self.final_scores[i] = scores
        self.final_scores = dict(sorted(self.final_scores.items(), key=lambda x: x[1], reverse=True))
    

    def WriteToFile(self):
        output_filename = os.path.join(OutputDir, f'{self.dataset}_bert.csv')
        with open(output_filename, 'w', newline='') as file:
            for i in self.final_scores:
                docid = i
                query = self.topics[self.qid]['title']
                rank = self.final_scores[i]
                file.write(f'Query ID: {self.qid} DocID: {docid} BERT score: {rank} Query: {query} ')
                    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Input dataset', type=str)

    args = parser.parse_args()
    if args.dataset:
        data =args.dataset 

    fetch = FetchText(data)
    fetch.ReadCSV()
    fetch.ModelEval()
    fetch.WriteToFile()
    qrels = get_qrels(THE_TOPICS[data])
    runfile = os.path.join(OutputDir, f'{data}_bert.csv')
    ndcg_score = evaluate(runfile, qrels, metric="ndcg_cut_10")
    print(f"nDCG@10 for {data}: {ndcg_score}")



