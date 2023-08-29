import torch
import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_labels: int):
        super(SoftmaxLoss, self).__init__()
        self.emb_dim = emb_dim
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.emb_dim*3, self.num_labels)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, features, labels):
        rep_a, rep_b = features

        vectors_concat = []
        vectors_concat.append(rep_a)
        vectors_concat.append(rep_b)
        vectors_concat.append(torch.abs(rep_a - rep_b))

        features = torch.cat(vectors_concat, dim=1)
        output = self.classifier(features)        
        loss = self.loss(output, labels.view(-1))
        return loss