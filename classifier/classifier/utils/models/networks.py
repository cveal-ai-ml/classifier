"""
Author: MINDFUL
Purpose: Deep Learning (DL) Models
"""


import torch
import torchvision
import lightning as L

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, Precision, Recall, F1Score


class Classifier(L.LightningModule):
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

        # Load: Network Parameters

        self.opti = params["optimizer"]
        self.num_epochs = params["num_epochs"]
        self.batch_size = params["batch_size"]
        self.learning_rate = params["learning_rate"]

        # Select: Network Architecture

        self.arch_id = params["arch"]
        self.num_classes = params["num_classes"]
        self.input_channels = params["sample_shape"][0]

        self.select_architecture()
        self.create_validation_measures()

    def create_validation_measures(self):
        """
        Creates confusion matrix measures for validation assessment
        """

        a = "macro"
        t = "multiclass"
        c = self.num_classes

        self.accuracy = Accuracy(task=t, num_classes=c)
        self.f1 = F1Score(task=t, average=a, num_classes=c)
        self.recall = Recall(task=t, average=a, num_classes=c)
        self.precision = Precision(task=t, average=a, num_classes=c)
        self.class_accuracy = Accuracy(task=t, num_classes=c, average=None)

    def select_architecture(self):
        """
        Load the DL model architecture
        """

        # Load: ConvNext Architecture

        if self.arch_id == 0:
            #self.name = "convnext"
            #weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
            #self.arch = torchvision.models.convnext_base(weights=weights)
            #self.arch = torchvision.models.convnext_base()
            #self.arch.features[0][0].in_channels = self.input_channels
            #self.arch.classifier[-1] = torch.nn.Linear(1024, self.num_classes)

            self.name = "EfficientNet-B2"
            weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
            self.arch = torchvision.models.efficientnet_b2(weights=weights)
            self.arch.classifier[-1] = torch.nn.Linear(1408, self.num_classes, bias=True)

        # Load: ResNext Architecture

        elif self.arch_id == 1:
            self.name = "resnext"
            weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
            self.arch = torchvision.models.resnext50_32x4d(weights=weights)
            self.arch.conv1.in_channels = self.input_channels
            self.arch.fc = torch.nn.Linear(2048, self.num_classes, bias=True)

        # Load: Vision Transformer Architecture

        elif self.arch_id == 2:
            #self.name = "vistransformer"
            #weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            #self.arch = torchvision.models.vit_b_16(weights=weights)
            #self.arch = torchvision.models.vit_b_16()
            #self.arch.conv_proj.in_channels = self.input_channels
            #self.arch.heads[-1] = torch.nn.Linear(768, self.num_classes)

            self.name = "ResNet50"
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.arch= torchvision.models.resnet50(weights=weights)
            self.arch.conv1.in_channels = self.input_channels
            self.arch.fc = torch.nn.Linear(2048, self.num_classes, bias=True)

        else:

            raise NotImplementedError

    def configure_optimizers(self):
        """
        Create DL learning rate optimizer and learning rate schedular
        """

        if self.opti == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), 
                                          lr=self.learning_rate)
        elif self.opti == "adam":
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.learning_rate)
        elif self.opti == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.learning_rate, momentum=0.9)
        else:
            raise NotImplementedError

        # lr_scheduler = ReduceLROnPlateau(optimizer, "min")
        # lr_scheduler_config = {"scheduler": lr_scheduler,
        #                        "monitor": "valid_error_epoch",
        #                        "interval": "epoch", "frequency": 1}

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def objective(self, preds, labels):
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

    def training_step(self, batch, batch_idx):
        """
        Run iteration for training dataset

        Parameters:
        - batch (tuple[torch.tensor[float]]): dataset mini-batch
        - batch_idx (int): index of current mini-batch

        Returns:
        - (torch.tensor[float]): Mini-batch loss
        """

        samples, labels, _ = batch

        # Gather: Predictions

        preds = self.arch(samples)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("train_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Run iteration for validation dataset

        Logs objective loss and confusion matrix measures

        Parameters:
        - batch (tuple[torch.tensor[float]]): dataset mini-batch
        - batch_idx (int): index of current mini-batch
        """

        samples, labels, _ = batch

        # Gather: Predictions

        preds = self.arch(samples)

        # Calculate: Objective Loss

        loss = self.objective(preds, labels)

        self.log("valid_error", loss, batch_size=self.batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        # Calculate: Confusion Matrix Analytics

        preds = torch.argmax(preds, dim=1)

        measures = {"accuracy": self.accuracy, "f1": self.f1,
                    "recall": self.recall, "precision": self.precision}

        for current_key in measures.keys():
            score = measures[current_key](preds, labels)
            self.log(current_key, score, batch_size=self.batch_size,
                     on_step=True, on_epoch=True, sync_dist=True)

        for i in range(self.num_classes):
            score = self.class_accuracy(preds, labels)
            self.log("class_%s_acc" % i, score[i], 
                     batch_size=self.batch_size,
                     on_step=True, on_epoch=True, sync_dist=True)

    def forward(self, samples):
        """
        Run DL forward pass for testing prediction / inference

        Parameters:
        - samples (torch.tensor[float]): data input as batch

        Returns:
        - (torch.tensor[float]): DL predictions
        """

        return self.arch(samples)
