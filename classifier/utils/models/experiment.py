"""
Author: MINDFUL
Purpose: Run Deep Learning (DL) training & validation or testing experiment
"""


import torch

from utils.models.logger import log_exp
from utils.misc.general import create_folder
from utils.models.networks import Classifier
from utils.data.prepare import load_train, load_test
from utils.models.support import config_hardware, load_progress


def testing(model, datasets, params):
    """
    Create testing procedure

    Parameters:
    - model (Classifier): deep learning model
    - datsaets (torch.utils.data.DataLoader): relevant datasets
    - params (dict[str, any]): network parameters
    """

    # Gather: Prediction Method
    # - Predictions for distributed parallel model requires unique referencing

    if hasattr(model, "module"):
        cycle = model.module.test_cycle
    else:
        cycle = model.test_cycle

    # Gather: Testing Dataset
    # - For deployment, there is only the testing dataset.
    # - Otherwise, select only the testing dataset

    if isinstance(datasets, list):
        _, _, test_data = datasets
    else:
        test_data = datasets

    # Testing: Model

    model.eval()
    cycle(test_data, model.rank)


def train_and_validate(model, datasets, params):
    """
    Create training & validation procedure

    Parameters:
    - model (Classifier): deep learning model
    - datsaets (torch.utils.data.DataLoader): relevant datasets
    - params (dict[str, any]): network parameters
    """

    # Gather: Prediction Method
    # - Predictions for distributed parallel model requires unique referencing

    if hasattr(model, "module"):
        start = model.module.epoch
        cycle = model.module.epoch_cycle
    else:
        start = model.epoch
        cycle = model.epoch_cycle

    # Gather: Training & Validation Datasets

    train_data, valid_data, _ = datasets

    # Run: Training & Validation

    for epoch in range(start, params["num_epochs"]):

        # - Run training cycle

        cycle(train_data, "train", model.rank)

        # - Run validation cycle

        if epoch % params["valid_rate"] == 0:

            model.eval()

            cycle(valid_data, "valid", model.rank)

            model.train()


def select_network(network_params, system_params):
    """
    Configure DL model for experiment

    Parameters:
    - network_params (dict[str, any]): network parameters
    - system_params (dict[str, any]): system parameters

    Returns:
    - (Classifier): DL model
    """

    # Load: Network

    model = Classifier(network_params)
    model, rank = config_hardware(model, system_params["gpus"])
    model.rank = rank

    # Selection: Start Over || Continue

    if network_params["use_progress"]:
        model = load_progress(model, network_params["path_network"],
                              system_params["gpus"], rank)

    # Update: Model Rank
    # - Reminder this updates the data structure in memory

    network_params["rank"] = rank

    return model


def seed_everything(params):
    """
    Create seed for experiment reproducability

    Parameters:
    - params (dict[str, any]): system parameters
    """

    if params["seed"]["flag"]:

        torch.manual_seed(params["seed"]["sequence"])
        torch.cuda.manual_seed(params["seed"]["sequence"])

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def run(params):
    """
    Run experiment for DL model (training and validation, testing)

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    # Set: Global Randomization Seed

    seed_everything(params["system"])

    # Selection: Testing Pipeline

    if int(params["network"]["deploy"]):

        # - Load DL model

        params["network"]["use_progress"] = 1
        model = select_network(params["network"], params["system"])

        # - Display experiment configuration

        if params["network"]["rank"] == 0:

            print("\nTesting Experiment")

            params["paths"]["train"] = None
            params["paths"]["valid"] = None
            log_exp(params["paths"], params["system"]["gpu_config"])

        # - Load testing dataset

        test = load_test(params["dataset"])

        # - Run testing procedure

        if params["network"]["rank"] == 0:
            print("\n-------------------------\n")
            print("Experiment Progress (Testing):\n")

        testing(model, test, params["network"])

        if params["network"]["rank"] == 0:
            print("\nTesting Procedure Complete\n")

    else:

        # - Load DL model

        model = select_network(params["network"], params["system"])

        # - Display experiment configuration

        if params["network"]["rank"] == 0:

            print("\nTraining Experiment")

            log_exp(params["paths"], params["system"]["gpus"])

            create_folder(params["network"]["path_results"],
                          not params["network"]["use_progress"])

        # - Load training, validation, and testing datasets

        datasets = load_train(params["dataset"])

        # - Run training procedure

        if params["network"]["rank"] == 0:
            print("\n-------------------------\n")
            print("Experiment Progress (Training):\n")

        train_and_validate(model, datasets, params["network"])

        # - Run testing procedure

        if params["network"]["rank"] == 0:
            print("\n-------------------------\n")
            print("Experiment Progress (Testing):\n")

        testing(model, datasets, params["network"])

        if params["network"]["rank"] == 0:
            print("\nTraining Procedure Complete\n")
