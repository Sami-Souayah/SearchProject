import pyserini
from indexpaths import THE_INDEX,THE_TOPICS
import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels


class FetchText():
    def __init__(self,dataset):
        self.dataset = dataset
        self.text = {}

        
    
    def FetchText(self, query, docid):
        searcher =  LuceneSearcher.from_prebuilt_index(THE_INDEX[self.dataset])
        doc = searcher.doc(docid)
        json_doc = json.loads(doc.raw())
        if self.dataset == 'dl19' or self.dataset == 'dl20':
            self.text[query] = json_doc['contents']
        else:
            self.text[query] = json_doc['text']


if __name__ == "__main__":
    dataset = 'fiqa'
    searcher =  LuceneSearcher.from_prebuilt_index(THE_INDEX[dataset])
    doc = searcher.doc('580802')
    print(doc.raw())

