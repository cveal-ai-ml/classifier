"""
Author: MINDFUL
Purpose: Deep Learning (DL) Experiment Logger
"""


import os
import pickle
import pandas as pd


def log_predict(results, path):
    """
    Log DL prediction matrix

    Parameters:
    - results (torch.tensor[float]): DL predictions
    - path (str): path to save data
    """

    path_file = os.path.join(path, "test_preds.pkl")

    with open(path_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def log_train_valid(results, iteration, epoch, i,
                    total, path, title, rate, name):
    """
    Log training and validation error

    Parameters:
    - results (torch.tensor[float]): training, validation error
    - iteration (int): global current training or validation batch iteration
    - epoch (int): global current epoch
    - i (int): local current training or validation batch iteration
    - total (int): number of local training batches for one epoch
    - path (str): path to save data
    - title (str): signifier to separate training and validation
    - rate (int): log rate for viewing training and validation analytics
    - name (str): DL model name
    """

    # Display: Current Progress

    results["iteration"] = iteration
    results["epoch"] = epoch

    if iteration % rate == 0:

        print("%s results (%s / %s):" % (title, i, total), end=" ")

        for j, current_key in enumerate(results.keys()):

            end = "\n" if j == len(results.keys()) - 1 else " | "

            value = round(results[current_key], 4)
            print("%s = %s" % (current_key, value), end=end)

    # Update: Total Progress

    results["model"] = name

    path_file = os.path.join(path, title + "_loss.csv")

    results = pd.DataFrame([results])

    if not os.path.exists(path_file):
        results.to_csv(path_file, index=False)
    else:
        data_file = pd.read_csv(path_file)
        data_file = pd.concat((data_file, results))
        data_file.to_csv(path_file, index=False)


def log_exp(path_params, system_params):
    """
    View experiment details

    Parameters:
    - path_params (dict[str, any]): path parameters
    - system_params (dict[str, any]): system parameters
    """

    print("\n-------------------------\n")

    print("Experiment: System\n")
    for current_key in system_params.keys():
        print("%s : %s" % (current_key, system_params[current_key]))

    print("\n-------------------------\n")

    print("Experiment: Paths\n")
    for current_key in path_params.keys():
        print("%s : %s" % (current_key, path_params[current_key]))
