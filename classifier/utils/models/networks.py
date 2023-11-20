"""
Author: MINDFUL
Purpose: Deep Learning (DL) Models
"""


import torch
import torchvision
import pytorch_warmup

from torch.optim.lr_scheduler import ExponentialLR

from utils.models.support import save_progress
from utils.models.logger import log_train_valid, log_predict


class Classifier(torch.nn.Module):
    """
    DL classification network using pre-trained DL models
    """

    def __init__(self, params):
        """
        Setup the DL classifier

        Parameters:
        - params (dict[str, any]): user defined network parameters
        """

        super().__init__()

        # Load: Path Paramters

        self.path = params["path_results"]

        # Load: Dataset Parameters

        self.learning_rate = params["learning_rate"]
        self.epochs = params["num_epochs"]
        self.log_rate = params["log_rate"]
        self.iteration = 0
        self.epoch = 0

        # Define: System Rank

        self.rank = None

        # - Select architecture

        self.select_architecture(params["arch"],
                                 params["sample_shape"][0],
                                 params["num_classes"])

        # - Initialize optimizer

        self.configure_optimizer()

        # - Initialize learning rate schedular

        self.configure_schedular()

    def select_architecture(self, choice, in_channels, num_classes):
        """
        Load the DL model architecture

        Parameters:
        - choice (int): pre trained network architecture
        - in_channels (int): number of image input channel
        - num_classes (int): number of classes
        """

        # Load: ConvNext Architecture

        if choice == 0:
            self.name = "convnext"
            weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
            self.arch = torchvision.models.convnext_base(weights=weights)
            self.arch.features[0][0].in_channels = in_channels
            self.arch.classifier[-1] = torch.nn.Linear(1024, num_classes)

        # Load: ResNext Architecture

        elif choice == 1:
            self.name = "resnext"
            weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
            self.arch = torchvision.models.resnext50_32x4d(weights=weights)
            self.arch.conv1.in_channels = in_channels
            self.arch.fc = torch.nn.Linear(2048, num_classes)

        # Load: Vision Transformer Architecture

        elif choice == 2:
            self.name = "vistransformer"
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            self.arch = torchvision.models.vit_b_16(weights=weights)
            self.arch.conv_proj.in_channels = in_channels
            self.arch.heads[-1] = torch.nn.Linear(768, num_classes)

        else:

            raise NotImplementedError

    def classi_loss(self, preds, labels):
        """
        Invoke classification objective function

        Parameters:
        - preds (torch.tensor[float]): model predictions
        - labels (torch.tensor[float]): human truth annotations

        Returns:
        - (torch.tensor[float]): classification error
        """

        obj = torch.nn.CrossEntropyLoss()

        return obj(preds, labels)

    def objective(self, preds, labels):
        """
        Invoke objective function

        Parameters:
        - preds (torch.tensor[float]): model predictions
        - labels (torch.tensor[float]): human truth annotations

        Returns:
        - (dict[str, float]): objective error
        """

        classi = self.classi_loss(preds, labels)

        return {"total": classi}

    def configure_schedular(self):
        """
        Create DL learning rate schedular
        """

        self.warmup = pytorch_warmup.UntunedLinearWarmup(self.optimizer)
        self.post_warmup = ExponentialLR(self.optimizer, gamma=0.4)

    def configure_optimizer(self):
        """
        Create DL optimization function
        """

        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           lr=self.learning_rate)

    def forward(self, x):
        """
        Run DL forward pass

        Parameters:
        - x (torch.tensor[float]): data input as batch

        Returns:
        - (torch.tensor[float]): DL predictions
        """

        return self.arch(x)

    def test_cycle(self, data, rank):
        """
        Run epoch cycle for DL testing

        Parameters:
        - data (torch.tensor[float]): testing dataset
        - rank (rank[int]): parallel process id

        Returns:
        - (torch.tensor[float]): DL predictions
        """

        device = next(self.parameters()).device

        results = {"preds": [], "truths": [], "filenames": []}

        for i, batch in enumerate(data):

            samples, labels, filenames = batch

            samples = samples.to(device)

            # - Evaluate dataset samples

            preds = self(samples).detach().to("cpu")

            results["preds"].append(preds)
            results["truths"].append(labels)
            results["filenames"].append(filenames)

        if rank == 0:
            log_predict(results, self.path_save)

    def epoch_cycle(self, data, title, rank):
        """
        Run epoch cycle for DL training and validation

        Parameters:
        - data (torch.tensor[float]): relevant dataset
        - title (str): signifier for training or validation
        - rank (rank[int]): parallel process id

        Returns:
        - (torch.tensor[float]): DL predictions
        """

        device = next(self.parameters()).device

        for i, batch in enumerate(data):

            samples, labels, _ = batch

            samples = samples.to(device)
            labels = labels.to(device)

            # - Evaluate dataset samples

            if title == "train":
                self.optimizer.zero_grad()

            preds = self(samples)

            # - Calculate objective performance

            loss = self.objective(preds, labels)

            # - Update: Network Parameters

            if title == "train":
                loss["total"].backward()
                self.optimizer.step()
                with self.warmup.dampening():
                    pass

            # - Track objective performance

            for current_key in loss.keys():
                loss[current_key] = loss[current_key].item()

            if rank == 0:
                log_train_valid(loss, self.iteration, self.epoch,
                                i, len(data), self.path, title,
                                self.log_rate, self.name)

                save_progress(self, self.path)

            self.iteration += 1

        self.epoch += 1

        with self.warmup.dampening():
            self.post_warmup.step()
