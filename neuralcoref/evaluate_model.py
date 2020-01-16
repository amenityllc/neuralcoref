import os
import sys
import spacy
from types import SimpleNamespace

from neuralcoref import NeuralCoref
from train.dataset import NCDataset
from train.evaluator import ConllEvaluator


PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PACKAGE_DIRECTORY, 'train'))


ALL_MENTIONS_PATH = os.path.join(PACKAGE_DIRECTORY, "test_mentions.txt")
DATA_PATH = '/Users/staveshemesh/Projects/Information-Extraction/Coref-Resolution/conll-2012/train/numpy/' #tmp
EVAL_DATA_PATH = '/Users/staveshemesh/Projects/Information-Extraction/Coref-Resolution/conll-2012/test/' #tmp

nlp = spacy.load('en')
coref = NeuralCoref(nlp.vocab)
model = coref.model

# Imitate learn.py's command line argument
args = SimpleNamespace(**{
    "costs": {"FN": 0.8, "FL": 0.4, "WL": 1.0},
    "lazy": True,
    "cuda": False,
    "batchsize": 20000, # Size of a batch in total number of pairs
    "numworkers": 8, # Number of workers for loading batches
})

test_key_file = f'{EVAL_DATA_PATH}/key.txt'
dataset = NCDataset(DATA_PATH, args)
embed_path = None
test_data_path = f'{EVAL_DATA_PATH}/numpy/'

evaluator = ConllEvaluator(model, dataset, test_data_path, test_key_file, embed_path, args)
evaluator.build_test_file(out_path=ALL_MENTIONS_PATH, print_all_mentions=True)
evaluator.get_score(ALL_MENTIONS_PATH, debug=True)
