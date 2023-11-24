"""
Author: MINDFUL
Purpose: General support utilities for DL models
"""


import os
import torch
import pickle
import numpy as np

from tqdm import tqdm

from utils.misc.general import create_folder
from utils.models.networks import Classifier


def model_inference(data, model, accelerator, path_save):
    """
    Run inference using trained model on dataset

    This currenly supports only 1 GPU

    Parameters:
    - data (torch.utils.data.DataLoader): infernce dataset
    - model (Classifier): trained DL model
    - accelerator (str): tag for device acceleration
    - path_save (str): path to save inference results
    """

    # Run: Model Inference

    model = model.to(accelerator)

    results = {"preds": [], "truths": [], "filenames": []}

    for batch in tqdm(data, desc="Model Inference"):

        # - Gather batch details

        samples, labels, filenames = batch
        samples = samples.to(accelerator)

        # - Evaluate dataset samples

        with torch.no_grad():
            preds = model(samples).detach()

        # - Organize results

        results = {}
        results["preds"].append(preds)
        results["truths"].append(labels)
        results["filenames"].append(filenames)

    # Save: Inference Results

    path_folder = os.path.join(path_save, "lightning_logs", "testing")
    create_folder(path_folder, overwrite=True)

    path_file = os.path.join(path_folder, "inference.pkl")
    with open(path_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(params, pre_trained):
    """
    Create deep learning model and optionally load pre-trained history

    Parameters:
    - params: user defined parameters
    - pre_trained: flag for enabling pretrained history

    Returns:
    (Classifier): DL model
    """

    path_root = os.path.join(params["paths"]["results"],
                             "lightning_logs", "training")

    model = Classifier(params["network"])

    # Load: Pre-Trained Model Parameters

    if pre_trained:
        path_params = os.path.join(path_root, "hparams.yaml")
        path_check = os.path.join(path_root, "checkpoints")

        # - Gather the most recent model
        # -- This string parses the filenames and sorts them in ascending order

        all_options = np.asarray([os.path.join(path_check, ele)
                                  for ele in os.listdir(path_check)])

        all_tags = [int(ele.strip(".ckpt"))
                    for current_value in all_options
                    for ele in current_value.split("=")
                    if ele.strip(".ckpt").isnumeric()]

        indices = np.argsort(all_tags)
        path_model = all_options[indices][-1]

        # - Load the history files

        model = model.load_from_checkpoint(checkpoint_path=path_model,
                                           hparams_file=path_params)

    return model
