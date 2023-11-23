"""
Author: MINDFUL
Purpose: Create a set of misc library specific tools
"""


import os
import sys
import torch


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

    print("\n-------------------------")

def update_network_params(params):
    """
    Update network parameters

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    params["network"]["sample_shape"] = params["dataset"]["sample_shape"]
    params["network"]["num_classes"] = params["dataset"]["num_classes"]
    os.environ["TORCH_HOME"] = params["paths"]["pre_trained_models"]


def update_path_params(params):
    """
    Update path parameters

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    # Update: Path Results

    root_name = str(params["network"]["arch"])
    path_results = os.path.join(params["paths"]["results"], root_name)
    params["paths"]["results"] = path_results


def update_dataset_params(params):
    """
    Update dataset parameters

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    params["dataset"]["path_train"] = params["paths"]["train"]
    params["dataset"]["path_valid"] = params["paths"]["valid"]
    params["dataset"]["path_test"] = params["paths"]["test"]

    params["dataset"]["num_workers"] = params["system"]["num_workers"]
    params["dataset"]["batch_size"] = params["network"]["batch_size"]

def update_system_params(params):
    """
    Update system parameters and validate GPU acceleration

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    num_devices = "auto"
    accelerator = "cpu"
    strategy = "auto"

    if params["system"]["gpus"]["enable"]:

        # Option: MacOS Silicon GPU

        if sys.platform == "darwin" and torch.backends.mps.is_available():
            num_devices = 1
            accelerator = "mps"

        # Option: NVIDIA GPU

        else:
            if "num_gpus" in params["system"]["gpus"].keys():
                num_devices = params["system"]["gpus"]["num_devices"]
            else:
                num_devices = torch.cuda.device_count()

            accelerator = "cuda"
            strategy = "ddp"

    # Update: System Parameters

    params["system"]["gpus"]["strategy"] = strategy
    params["system"]["gpus"]["platform"] = sys.platform
    params["system"]["gpus"]["num_devices"] = num_devices
    params["system"]["gpus"]["accelerator"] = accelerator


def update_params(params):
    """
    Update parameters so each parameter group contains appropriate information

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    # Update: System Configuration

    update_system_params(params)

    # Update: Dataset Configuration

    update_dataset_params(params)

    # Update: Path Configuration

    update_path_params(params)

    # Update: Network Config

    update_network_params(params)


def override_params(params):
    """
    Override specific parameters using command line arguments

    Parameters:
    - params (dict[str, any]): YAML parameters
    - args (dict[str, any]): CL parameters
    """

    args = params["cl"]

    arg_list = list(args.keys())

    # Override: Path (Input / Output) Parameters

    if "train" in arg_list:
        params["paths"]["train"] = args["train"]

    if "valid" in arg_list:
        params["paths"]["valid"] = args["valid"]

    if "test" in arg_list:
        params["paths"]["test"] = args["test"]

    if "results" in arg_list:
        params["paths"]["results"] = args["results"]

    if "cache" in arg_list:
        params["paths"]["pre_trained_models"] = args["cache"]

    # Override: Dataset Parameters

    if "batch_size" in arg_list:
        params["dataset"]["batch_size"] = int(args["batch_size"])

    # Override: DL Parameters

    if "deploy" in arg_list:
        params["network"]["deploy"] = args["deploy"]

    if "arch" in arg_list:
        params["network"]["arch"] = int(args["arch"])

    # Override: System Parameters

    if "use_gpu" in arg_list:
        params["system"]["gpus"]["enable"] = int(args["use_gpu"])

    if "num_gpus" in arg_list:
        params["system"]["gpus"]["num_devices"] = int(args["num_devices"])

    if "num_workers" in arg_list:
        params["system"]["num_workers"] = int(args["num_workers"])
