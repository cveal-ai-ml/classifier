"""
Author: MINDFUL
Purpose: Create dashboard utilities
"""

import os
import numpy as np
import pandas as pd


def median_filter(all_data, v_tag="total", t_tag="iteration", step_size=20):

    new_data = {}

    for current_key in all_data.keys():

        time = all_data[current_key][t_tag]
        measures = all_data[current_key][v_tag]

        new_time, new_measures = [], []

        i = 0
        while i < len(time):
            t_group = time[i:i + step_size]
            m_group = measures[i:i + step_size]

            new_time.append(np.median(t_group))
            new_measures.append(np.median(m_group))

            i += step_size

        new_data[current_key] = {v_tag: new_measures, t_tag: new_time}

    return new_data


def load_loss(path, tag):

    all_model_folders = os.listdir(path)
    all_model_folders.sort()

    all_data = {}
    for model_folder in all_model_folders:
        path_load = os.path.join(path, model_folder)

        if tag == "train":
            path_data = os.path.join(path_load, "train_loss.csv")
        else:
            path_data = os.path.join(path_load, "valid_loss.csv")

        data = pd.read_csv(path_data)
        tag = data["model"][0]
        all_data[tag] = data

    return all_data
