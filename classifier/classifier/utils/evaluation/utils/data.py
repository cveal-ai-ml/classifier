"""
Author: MINDFUL
Purpose: Create dashboard utilities
"""

import os
import numpy as np
import pandas as pd


def median_filter(measures, time, step_size=100):

    new_time, new_measures = [], []

    i = 0
    while i < len(time):
        t_group = time[i:i + step_size]
        m_group = measures[i:i + step_size]

        new_time.append(np.median(t_group))
        new_measures.append(np.median(m_group))

        i += 1

    return (new_measures, new_time)


def format_data(df, v_tag, t_tag, use_filter=0):

    df = df[df[v_tag].notna()]
    measures, time = df[v_tag], df[t_tag]

    if use_filter == 1:
        if use_filter == 2:
            step_size = 1167
            is_epoch = 1
        else:
            step_size = 100

        measures, time = median_filter(measures, time,
                                       step_size=step_size)

    if "lr" in v_tag:
        time = np.arange(len(measures))

    return (measures, time)


def load_loss(path):

    tag = "lightning_logs/training/metrics.csv"

    all_model_folders = os.listdir(path)
    all_model_folders.sort()

    all_data = {}
    for i, model_folder in enumerate(all_model_folders):
        if model_folder == "slurm":
            continue
        path_data = os.path.join(path, model_folder, tag)
        data = pd.read_csv(path_data)
        all_data["Model %s" % model_folder] = data

    return all_data
