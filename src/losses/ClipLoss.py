import torch
import torch.nn as nn


class ClipLoss(nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, features, labels):
        rep_a, rep_b = features
        sim = torch.mm(rep_a, rep_b.transpose(0, 1))
        loss = (self.loss(sim, labels) + self.loss(sim.transpose(0, 1), labels)) / 2

        return loss