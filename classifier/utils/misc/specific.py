"""
Author: MINDFUL
Purpose: Create a set of misc library specific tools
"""


import os
import sys
import torch


def update_network_params(params):
    """
    Update network parameters

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

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

    # Gather: DL Model History
    # - History being past model training files

    history = []
    if os.path.exists(path_results):
        for data_file in os.listdir(path_results):
            if ".checkpoint" in data_file:
                history.append(data_file)

    history.sort()

    # Update: Transfer Learning / Training Continuation

    continue_training = params["network"]["use_progress"]

    if len(history) != 0 and continue_training:
        path_history = os.path.join(path_results, history[-1])
        params["paths"]["network"] = path_history
        params["network"]["path_network"] = path_history
    else:
        params["network"]["path_network"] = ""

    # Update: Addtional Necessities

    params["network"]["sample_shape"] = params["dataset"]["sample_shape"]
    params["network"]["path_results"] = path_results


def update_dataset_params(params):
    """
    Update dataset parameters

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    params["dataset"]["path_train"] = params["paths"]["train"]
    params["dataset"]["path_valid"] = params["paths"]["valid"]
    params["dataset"]["path_test"] = params["paths"]["test"]


def update_system_params(params):
    """
    Update system parameters and validate GPU acceleration

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    gpu_enabled = params["system"]["gpus"]["enable"]

    # Update: MacOS

    if sys.platform == "darwin":
        flag = torch.backends.mps.is_available()
        name = "MacOS M-Series"

        num_gpus_per_node = 0
        total_num_gpus = 0
        num_nodes = 1

        if flag and gpu_enabled:
            num_gpus_per_node = 1
            total_num_gpus = 1

    # Update: Linux & Windows

    else:

        if gpu_enabled:
            name = "CUDA"
            total_num_gpus = os.environ["WORLD_SIZE"]
            num_gpus_per_node = torch.cuda.device_count()
            num_nodes = int(total_num_gpus) // int(num_gpus_per_node)
        else:
            name = "CPU"
            num_nodes = 1
            total_num_gpus = 0
            num_gpus_per_node = 1

    # Update: System Parameters

    params["system"]["gpus"]["device"] = name
    params["system"]["gpus"]["num_nodes"] = num_nodes
    params["system"]["gpus"]["platform"] = sys.platform
    params["system"]["gpus"]["total_num_gpus"] = total_num_gpus
    params["system"]["gpus"]["num_gpus_per_node"] = num_gpus_per_node


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

    # Override: Dataset Parameters

    if "batch_size" in arg_list:
        params["dataset"]["batch_size"] = int(args["batch_size"])

    # Override: DL Parameters

    if "deploy" in arg_list:
        params["network"]["deploy"] = args["deploy"]

    if "arch" in arg_list:
        params["network"]["arch"] = int(args["arch"])

    if "use_gpu" in arg_list:
        params["system"]["gpus"]["enable"] = int(args["use_gpu"])
