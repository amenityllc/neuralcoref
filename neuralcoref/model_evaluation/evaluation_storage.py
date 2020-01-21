import pandas as pd


OUTPUT_COLS = [
    # experiment details
    'dataset',
    'test_file',
    'time',

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

class ScoringResultStorage:
    def __init__(self, score, timestamp):
        self.scoring_result = dict.fromkeys(OUTPUT_COLS)
        self.evaluator_score = score
        self.timestamp = timestamp

    def build_result(self):
        # populate metdata
        self.scoring_result['dataset'] = DATASET_NAME
        self.scoring_result['test_file'] = ALL_MENTIONS_PATH
        self.scoring_result['time'] = time.strptime(timestr, "%Y%m%d-%H%M%S")

        # populate scores
        score, F1_conll, ident = self.sevaluator_scorecore

        for metric in ['muc', 'bcub', 'ceafe']:
            for measure_name, measure_score  in zip(['precision', 'recall', 'f1'], score[metric]):
                self.scoring_result[f'{metric}_{measure_name}'] = measure_score

        self.scoring_result['F1_conll'] = F1_conll


    def store_result(self):
        df_result = pd.DataFrame(self.scoring_result)
        df_result.to_csv('df_output')

