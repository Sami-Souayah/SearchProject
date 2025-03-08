import pyserini
from transformers import AutoTokenizer
import torch
from indexpaths import THE_INDEX,THE_TOPICS
import json
import csv
import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

directory = '/Users/sami/Desktop/MIT Research Project/CSV Files'
class FetchText():
    def __init__(self,dataset):
        self.dataset = dataset
        self.text = {}
        self.tokenized_text = {}
        self.topics = get_topics(THE_TOPICS[dataset] if dataset != 'dl20' else 'dl20')
        self.directory = f"/Users/sami/Desktop/MIT Research Project/CSV Files/{dataset}_run.csv"
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")
    
    def FetchText(self, docid):
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
        self.tokenized_text[docid] = self.tokenize_text(text)
    def ReadCSV(self):
        with open(self.directory, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=' ')
            for i,row in enumerate(reader):
                if i == 10:
                    break
                docID = row[2]
                self.FetchText(docID)
    def tokenize_text(self, text, max_length=512):
        encoding = self.tokenizer(
            text,
            padding="max_length", 
            truncation=True,  
            max_length=max_length, 
            return_tensors="pt", 
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }



if __name__ == "__main__":
    fetch = FetchText('fiqa')
    fetch.ReadCSV()
    print(fetch.tokenized_text)
