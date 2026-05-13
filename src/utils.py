from rich.console import Console
import re
import pandas as pd
import json

# Helper functions
def results_to_df(results, verbose_logs=False):

    if verbose_logs:
        eval_df_ = pd.DataFrame([
            {
                'input': i.test_results[0].input,
                'polarity_match': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Polarity Match"],
                'precision': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Precision"],
                'recall': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Recall"],
                'f1_score': i.test_results[0].metrics_data[0].score,
                'reason': i.test_results[0].metrics_data[0].reason,
                'verbose_logs': json.dumps(json.loads(i.test_results[0].metrics_data[0].verbose_logs), indent=2),
                'key':i.test_results[0].additional_metadata['key']
            }
            for i in results
        ])

    else:
        eval_df_ = pd.DataFrame([
            {
                'input': i.test_results[0].input,
                'polarity_match': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Polarity Match"],
                'precision': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Precision"],
                'recall': json.loads(i.test_results[0].metrics_data[0].verbose_logs)["Recall"],
                'f1_score': i.test_results[0].metrics_data[0].score,
                'reason': i.test_results[0].metrics_data[0].reason,
                'key':i.test_results[0].additional_metadata['key']
            }
            for i in results
        ])

    return eval_df_


'''
Silent console context
'''
class SilentConsole:
    def __enter__(self):
        # Save original methods
        self._orig_print = Console.print
        self._orig_log = Console.log
        # Patch to silence
        Console.print = lambda *args, **kwargs: None
        Console.log = lambda *args, **kwargs: None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods
        Console.log = self._orig_log
        Console.print = self._orig_print
