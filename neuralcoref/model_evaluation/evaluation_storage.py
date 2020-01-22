import os
import json


OUTPUT_COLS = [
    # experiment details
    'dataset',
    'test_file',
    'timestamp',

    # scores
    'muc_precision',
    'muc_recall',
    'muc_f1',
    'bcub_precision',
    'bcub_recall',
    'bcub_f1',
    'ceafe_precision',
    'ceafe_recall',
    'ceafe_f1',
    'F1_conll'
]
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')


class ScoringResultStorage:
    @classmethod
    def save(cls, score, meta):
        storage = ScoringResultStorage(score, meta)
        storage.build_result()
        storage.store_result()

    def __init__(self, score, meta):
        self.scoring_result = dict.fromkeys(OUTPUT_COLS)
        self.evaluator_score = score
        self.meta = meta
        output_dir_name = f'{self.meta["timestr"]}_{self.meta["dataset"]}'
        self.output_path = os.path.join(RUNS_DIR, output_dir_name)

    def build_result(self):
        # populate metdata
        self.scoring_result['dataset'] = self.meta['dataset']
        self.scoring_result['test_file'] = self.meta['mentions_path']
        self.scoring_result['timestamp'] = self.meta['timestr']
        self.scoring_result['args'] = self.meta['args']

        # populate scores
        score, F1_conll, ident = self.evaluator_score

        for metric in ['muc', 'bcub', 'ceafe']:
            for measure_name, measure_score  in zip(['precision', 'recall', 'f1'], score[metric]):
                self.scoring_result[f'{metric}_{measure_name}'] = measure_score

        self.scoring_result['F1_conll'] = F1_conll


    def store_result(self):
        print('appending score result...')
        with open(f'{RUNS_DIR}/scores.txt', 'a') as fd:
            fd.write("\n"+json.dumps(self.scoring_result))
        print('done.')
