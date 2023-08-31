import torch
import torch.nn as nn


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()
        self.loss = nn.MSELoss()


    def forward(self, features, labels):
        rep_a, rep_b = features
        sim = torch.cosine_similarity(rep_a, rep_b)
        labels = labels if labels.dtype == torch.float else labels.float()
        loss = self.loss(sim, labels)
        
        return loss