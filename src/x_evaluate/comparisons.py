from typing import List, Dict

import pandas as pd

from x_evaluate.evaluation_data import EvaluationDataSummary, EvaluationData
from x_evaluate.trajectory_evaluation import create_trajectory_result_table_wrt_traveled_dist


def identify_common_datasets(summaries: Dict[str, EvaluationDataSummary]):
    common_datasets = None
    for s in summaries.values():
        if common_datasets is None:
            common_datasets = set(s.data.keys())
        common_datasets = common_datasets.intersection(s.data.keys())
    return common_datasets


def compare_trajectory_performance_wrt_traveled_dist(summaries: Dict[str, EvaluationDataSummary]) -> pd.DataFrame:
    common_datasets = identify_common_datasets(summaries)

    result_table = None

    for s in summaries.values():
        new_table = create_trajectory_result_table_wrt_traveled_dist(s)
        if result_table is None:
            result_table = new_table
        else:
            result_table = pd.merge(result_table, new_table, on="Dataset")

    return result_table



