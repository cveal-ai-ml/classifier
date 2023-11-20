"""
Author: MINDFUL
Purpose: Deep Learning (DL) support utilities
"""


import os
import torch
import pickle

from torch.nn.parallel import DistributedDataParallel as DDP


def config_hardware(model, params, rank=0):
    """
    Configure hardware for DL

    Parameters:
    - params (dict[str, any]): GPU parameters
    - rank (int): parallel process id

    Returns:
    - (tuple[any]): DL model and process rank
    """

    if params["enable"]:

        # Config: M1-MacOS GPU

        if params["platform"] == "darwin":
            model = model.to("mps")

        # Config: NVIDIA GPU(s)

        else:

            torch.distributed.init_process_group("nccl")
            rank = torch.distributed.get_rank()

            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])

    return (model, rank)


def save_progress(model, path):
    """
    Saving model training progress

    Parameters:
    - model (Classifier): deep learning system
    - path (str): path to saved deep learning model
    """

    iteration = model.iteration

    [os.remove(os.path.join(path, ele))
     for ele in os.listdir(path) if ".checkpoint" in ele]

    name = str(iteration) + ".checkpoint"

    params = {"iteration": model.iteration,
              "epoch": model.epoch}

    try:

        torch.save(model.state_dict(), os.path.join(path, name))
        pickle.dump(params, open(os.path.join(path, "params.pkl"), "wb"))

    except KeyboardInterrupt:

        torch.save(model.state_dict(), os.path.join(path, name))
        pickle.dump(params, open(os.path.join(path, "params.pkl"), "wb"))
        exit()


def load_progress(model, path, params, rank, tag=".checkpoint"):
    """
    Load trained DL model for transfer learning / training continuation

    Parameters:
    - model (Classifier): deep learning system
    - path (str): path to saved deep learning model
    - params (dict[str, any]): GPU parameters
    - rank (int): parallel process id

    Returns:
    - (Classifier): DL model
    """

    if os.path.exists(path):
        if params["platform"] == "darwin":
            history = torch.load(path)
        else:
            if params["enable"]:
                map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
                history = torch.load(path, map_location=map_location)
            else:
                history = torch.load(path)

        model.load_state_dict(history)

        filename = path.split("/")[-1]

        misc = path.replace(filename, "params.pkl")
        data = pickle.load(open(misc, "rb"))

        model.epoch = data["epoch"]
        model.iteration = data["iteration"]
        model.rank = rank

    return model
