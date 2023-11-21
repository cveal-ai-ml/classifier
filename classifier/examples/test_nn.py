

import torch


class NN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.arch = torch.nn.Conv2d(in_channels=3,
                                    out_channels=16,
                                    kernel_size=3)

        self.classi = torch.nn.Linear(16 * 30 * 30, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x):

        x = self.arch(x)
        x = x.view(x.size()[0], -1)

        return self.classi(x)

    def obj(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss()

        return obj(preds, labels)


if __name__ == "__main__":

    x = torch.rand(5, 3, 32, 32).cuda()
    y = torch.rand(5).round().long().cuda()

    model = NN().cuda()
    preds = model(x)

    model.optimizer.zero_grad()
    obj = model.obj(preds, y)
    obj.backward()
    model.optimizer.step()
