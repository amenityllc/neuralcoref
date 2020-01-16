import os
import sys
import spacy
from types import SimpleNamespace

from neuralcoref import NeuralCoref
from train.dataset import NCDataset
from train.evaluator import ConllEvaluator


sys.path.append((os.path.join(os.path.abspath(os.path.dirname(__file__)), 'train')))


DATA_PATH = '/Users/staveshemesh/Projects/Information-Extraction/Coref-Resolution/conll-2012/train/numpy/' #tmp
EVAL_DATA_PATH = '/Users/staveshemesh/Projects/Information-Extraction/Coref-Resolution/conll-2012/test/numpy/' #tmp

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
test_key_file = 'test_key'
dataset = NCDataset(DATA_PATH, args)
embed_path = None
evaluator = ConllEvaluator(model, dataset, EVAL_DATA_PATH, test_key_file, embed_path, args)
output_path = ''
evaluator.get_score('./output_score.txt', debug=True)


