"""
Author: MINDFUL
Purpose: Run Deep Learning (DL) training & validation or testing experiment
"""


import torch
import lightning as L

from lightning.pytorch.loggers import CSVLogger

from utils.misc.specific import log_exp
from utils.models.networks import Classifier
from utils.data.prepare import load_train, load_test


def test_experiment(params):
    """
    Run testing experiment for DL model

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    print("\nTesting Experiment\n")

    # Load: Testing Dataset

    test = load_test(params["dataset"])

    # Create: Model

    model = Classifier(params["network"])

    print("Experiment Progress (Testing):\n")


def train_experiment(params):
    """
    Run training & validation experiment for DL model

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    print("\nTraining Experiment\n")

    # Set: Global Randomization Seed

    seed_everything(params["system"])

    # Load: Datasets (Training, Validation)

    train, valid = load_train(params["dataset"])

    # Create: Model

    model = Classifier(params["network"])

    # Create: Logger

    exp_logger = CSVLogger(save_dir=params["paths"]["results"],
                           version="logs")

    # Create: Trainer

    num_epochs = params["network"]["num_epochs"]

    strategy = params["system"]["gpus"]["strategy"]
    num_devices = params["system"]["gpus"]["num_devices"]
    accelerator = params["system"]["gpus"]["accelerator"]

    trainer = L.Trainer(accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=20, logger=exp_logger)

    # Train: Model

    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)


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

    # Log: Experiment Setup

    log_exp(params["paths"], params["system"]["gpus"])

    # Selection: Experiment (Testing, Training)

    if int(params["network"]["deploy"]):
        test_experiment(params)
    else:
        train_experiment(params)

    print("\nExperiment Complete\n")
