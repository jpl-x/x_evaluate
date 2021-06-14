from typing import List

import numpy as np
import pandas as pd

from x_evaluate.evaluation_data import EvaluationDataSummary


def identify_common_datasets(summaries: List[EvaluationDataSummary]):
    common_datasets = None
    for s in summaries:
        if common_datasets is None:
            common_datasets = set(s.data.keys())
        common_datasets = common_datasets.intersection(s.data.keys())
    return common_datasets


def identify_changing_parameters(summaries: List[EvaluationDataSummary], filter=True, filter_keys=None,
                                 common_datasets=None):
    if not common_datasets:
        common_datasets = identify_common_datasets(summaries)

    if filter:
        if not filter_keys:
            # by default filter intitial states
            filter_keys = {'p', 'v', 'q', 'b_w', 'b_a'}

    params = [s.data[d].params for s in summaries for d in common_datasets]

    changing_parameters = get_changing_parameter_keys(params)

    if filter:
        return changing_parameters.difference(filter_keys)
    else:
        return changing_parameters


def create_parameter_changes_table(summaries: List[EvaluationDataSummary], common_datasets=None):
    changes = dict()
    equally_change_among_datasets = dict()

    for d in common_datasets:
        params = [s.data[d].params for s in summaries]

        changing_parameters = get_changing_parameter_keys(params)

        for k in changing_parameters:
            all_params = [p[k] for p in params]
            if k in changes:
                is_same = np.all([are_list_entries_equal(list(t)) for t in zip(changes[k], all_params)])
                if not is_same:
                    print(F"WARNING parameter '{k}' does not change consistently over datasets ("
                          F"{list(common_datasets)[0]}, {d})")
                    equally_change_among_datasets[k] = False
            else:
                changes[k] = all_params  # = changing_parameters.union({k})
                equally_change_among_datasets[k] = True

    data = {'Evaluation Runs': [s.name for s in summaries]}
    for k, v in changes.items():
        if equally_change_among_datasets[k]:
            data[k] = v
        else:
            data[F"{k} DIRTY"] = v

    parameter_changes_table = pd.DataFrame(data)
    parameter_changes_table = parameter_changes_table.T
    parameter_changes_table.columns = parameter_changes_table.iloc[0]
    parameter_changes_table = parameter_changes_table.iloc[1:]
    return parameter_changes_table


def get_changing_parameter_keys(params):
    common_keys = None
    changing_parameters = set()
    for p in params:
        if common_keys:
            diff = common_keys.difference(p)
            if len(diff) > 0:
                print(F"WARNING: different parameter types used when comparing parameters: {diff}")
            common_keys = common_keys.intersection(p.keys())
        else:
            common_keys = set(p.keys())
    for k in common_keys:
        all_params = [p[k] for p in params]
        if not are_list_entries_equal(all_params):
            changing_parameters.add(k)
    return changing_parameters


def are_list_entries_equal(all_params):
    if isinstance(all_params[0], list):
        are_all_equal = np.all(np.array(all_params) == all_params[0])
    else:
        are_all_equal = len(set(all_params)) <= 1
    return are_all_equal






