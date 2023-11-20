"""
Author: MINDFUL
Purpose: Example of learning schedulars

It requires installation of custom MIT package:
    - https://github.com/Tony-Y/pytorch_warmup/tree/master
    - pip install pytorch_warmup

When would you use this? When using ADAM like optimizer
"""


import torch
import pytorch_warmup
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR


plt.style.use("ggplot")


def plot_results(results, fig_size=(10, 6), font_size=12):
    """
    Plot Learning Rate Results

    Parameters:
    - all_lr (list[any]): learning rate values
    - fig_size (tuple[int]): figure size
    - font_size (int): font size
    """

    x_vals = torch.arange(len(results))

    fig, ax = plt.subplots(figsize=fig_size)

    ax.plot(x_vals, results)

    ax.set_xlabel("Time", fontsize=font_size)
    ax.set_ylabel("Learning Rate", fontsize=font_size)
    ax.set_title("Learning Rate Scheduling", fontsize=font_size)


if __name__ == "__main__":
    """
    Demonstrating warmup and annealing schedulars for DL learning rate
    """

    # Initialize: Parameters
    # - alpha: learning rate
    # - num_epochs: number of epochs
    # - num_iterations: number of batches per epoch

    alpha = 100
    num_epochs = 10
    num_iterations = 3000

    # Create: Arbitrary Model

    model = torch.nn.Linear(2, 1)

    # Create: Learning Rate Schedulars
    # - Warmup schedular starts small and peaks to the defined alpha
    # - Post warmup schedular minimizes the alpha value over time
    # -- "gamma" for this schedular controls the decay rate as it appraoches 1

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    warmup = pytorch_warmup.UntunedLinearWarmup(optimizer)
    post_warmup = ExponentialLR(optimizer, gamma=0.4)

    # Run: Example, Learning Rate Schedulars

    results = []
    for i in range(num_epochs):
        for i in range(num_iterations):
            optimizer.step()
            results.append(optimizer.param_groups[0]["lr"])

            # - Update the warmup schedular

            with warmup.dampening():
                pass

        # - Update the post warmup schedular

        with warmup.dampening():
            post_warmup.step()

    # Plot: Results

    plot_results(results)
    plt.show()
