"""
Author: MINDFUL
Purpose: Create dashboard utilities
"""

import os
import numpy as np
import pandas as pd


def median_filter(measures, time, step_size=20):

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

    if use_filter:
        measures, time = median_filter(measures, time)

    return (measures, time)


def load_loss(path):

    tag = "lightning_logs/training/metrics.csv"

    all_model_folders = os.listdir(path)
    all_model_folders.sort()

    all_data = {}
    for i, model_folder in enumerate(all_model_folders):
        path_data = os.path.join(path, model_folder, tag)
        data = pd.read_csv(path_data)
        all_data["Model %s" % i] = data

    return all_data
