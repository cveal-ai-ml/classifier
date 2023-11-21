

import torch
import torchvision


class NN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        self.arch = torchvision.models.convnext_base(weights=weights)
        self.arch.features[0][0].in_channels = 3
        self.arch.classifier[-1] = torch.nn.Linear(1024, 2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x):

        return self.arch(x)

    def obj(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss()

        return obj(preds, labels)


if __name__ == "__main__":

    x = torch.rand(5, 3, 224, 244).cuda()
    y = torch.rand(5).round().cuda()

    model = NN().cuda()
    preds = model(x)

    model.optimizer.zero_grad()
    obj = model.obj(preds, y)
    obj.backward()
    model.optimizer.step()
