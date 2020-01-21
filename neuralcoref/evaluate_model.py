import os
import sys
from types import SimpleNamespace
import time

from train.evaluator import ConllEvaluator
from train.utils import SIZE_EMBEDDING

from train.model import Model
from train.dataset import (
    NCDataset,
    NCBatchSampler,
    load_embeddings_from_file,
    padder_collate,
    SIZE_PAIR_IN,
    SIZE_SINGLE_IN,
)
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PACKAGE_DIRECTORY, 'train'))
timestr = time.strftime("%Y%m%d-%H%M%S")


ALL_MENTIONS_PATH = os.path.join(PACKAGE_DIRECTORY, f"{timestr}_eval.txt")
EVAL_DATA_PATH = '/Users/staveshemesh/Projects/Information-Extraction/Coref-Resolution/conll-2012/dev/' #tmp
EVAL_DATA_NUMPY_PATH = os.path.join(EVAL_DATA_PATH, 'numpy/')
EVAL_DATA_KEY_PATH = os.path.join(EVAL_DATA_PATH, 'key.txt')
EMBEDDING_PATH = os.path.join(PACKAGE_DIRECTORY, 'train/weights/')



# Imitate learn.py's command line argument
args = SimpleNamespace(**{
    "costs": {"FN": 0.8, "FL": 0.4, "WL": 1.0},
    "lazy": True,
    "cuda": False,
    "batchsize": 20000, # Size of a batch in total number of pairs
    "numworkers": 8, # Number of workers for loading batches
    "h1": 1000,  # Number of hidden unit on layer 1"
    "h2": 500, # Number of hidden unit on layer 2
    "h3": 500,  # Number of hidden unit on layer 3
})

dataset = NCDataset(EVAL_DATA_NUMPY_PATH, args)
tensor_embeddings, voc = load_embeddings_from_file(EMBEDDING_PATH + "tuned_word")

model = Model(
    len(voc),
    SIZE_EMBEDDING,
    args.h1,
    args.h2,
    args.h3,
    SIZE_PAIR_IN,
    SIZE_SINGLE_IN,
)
model.load_embeddings(tensor_embeddings)

evaluator = ConllEvaluator(model, dataset, EVAL_DATA_NUMPY_PATH, EVAL_DATA_KEY_PATH, None, args)
evaluator.build_test_file(out_path=ALL_MENTIONS_PATH,
                          remove_singleton=False,
                          print_all_mentions=False,
                          # debug=None)
                          debug=-1)
evaluator.get_score(ALL_MENTIONS_PATH, debug=True)
