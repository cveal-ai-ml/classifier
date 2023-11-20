"""
Author: MINDFUL
Purpose: Create a set of re-usable general tools
"""


import sys

from utils.models.experiment import run
from utils.misc.general import load_config
from utils.misc.specific import override_params, update_params


if __name__ == "__main__":

    """
    Load user parameters and run Deep Learning (DL) experiment
    Parameters are defined using Command Line (CL) & inside YAML
    """

    # Load: User Defined Parameters (YAML, CL)

    params = load_config(sys.argv)

    # Override: YAML Parameters Using CL

    override_params(params)

    # Update: Parameters

    update_params(params)

    # Run: DL Experiment

    run(params)
