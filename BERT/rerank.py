import pyserini
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
        self.topics = get_topics(THE_TOPICS[dataset] if dataset != 'dl20' else 'dl20')
        self.directory = f"/Users/sami/Desktop/MIT Research Project/CSV Files/{dataset}_run.csv"
    
    def FetchText(self, docid):
        searcher =  LuceneSearcher.from_prebuilt_index(THE_INDEX[self.dataset])
        doc = searcher.doc(docid)
        json_doc = json.loads(doc.raw())
        if self.dataset == 'dl19' or self.dataset == 'dl20':
            self.text[docid] = json_doc['contents']
        else:
            self.text[docid] = json_doc['text']
            if 'title' in json_doc:
                self.text[docid] = f"{json_doc['title']} {json_doc['text']}"
    def ReadCSV(self):
        with open(self.directory, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                docID = row[2]
                self.FetchText(docID)



if __name__ == "__main__":
    fetch = FetchText('fiqa')
    fetch.ReadCSV()
    print(fetch.text.keys())
