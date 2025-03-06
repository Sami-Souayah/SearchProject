import argparse
from BM25.indexpaths import THE_INDEX, THE_TOPICS
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help = 'Input dataset', type=str)
parser.add_argument('--k', help = 'Input k', type=int)

args = parser.parse_args()
if args.dataset:
    print(THE_INDEX[args.dataset])
if args.k:
    print(args.k)