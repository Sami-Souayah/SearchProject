import pyserini
import argparse
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from indexpaths import THE_INDEX,THE_TOPICS
import os


class BM25():

    def __init__(self, index_name, k1=0.82,k2=0.68):

        self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        self.searcher.set_bm25(k1, k2)
 

    def search(self, query, k=100):
        hits = self.searcher.search(query, k=k)
        return hits
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Input dataset', type=str)
    parser.add_argument('--k', help = 'Input k', type=int)

    args = parser.parse_args()
    if args.dataset:
        data =args.dataset 
    if args.k:
        k = args.k  
    bm25=BM25(index_name=THE_INDEX[data])
    topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
    qrels = get_qrels(THE_TOPICS[data])
    with open(f'{data}.csv', 'w', newline='') as file:
        for i in topics:
            query = topics[i]['title']
            qid = i
            hits = bm25.search(query, k=k)
            rank=1
            for hit in hits:
                file.write(f'QueryID: {qid} Document ID: {hit.docid} Rank: {rank} Score: {hit.score}\n')
                rank+=1
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m recall.100 {THE_TOPICS[data]} 'Stuff.csv'"))






        