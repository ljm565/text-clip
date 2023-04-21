import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel



# KoBERT
class BERT(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(BERT, self).__init__()
        self.tokenizer = tokenizer
        self.device = device

        self.model = BertModel.from_pretrained('skt/kobert-base-v1')

        self.hidden_dim = self.model.config.hidden_size

        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=config.layernorm_eps)
        self.src_wts = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trg_wts = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def make_mask(self, x):
        mask = torch.where(x==self.tokenizer.pad_token_id, 0, 1)
        return mask

    
    def find_eos(self, x):
        mask = torch.tensor(x == self.tokenizer.sep_token_id, dtype=torch.long)
        mask = torch.argmax(mask, dim=1)
        return mask
        

    def forward(self, src, trg):
        batch_size = src.size(0)
        src_mask, trg_mask = self.make_mask(src), self.make_mask(trg)
        src_eos, trg_eos = self.find_eos(src), self.find_eos(trg)

        src, trg = self.model(input_ids=src, attention_mask=src_mask)['last_hidden_state'], self.model(input_ids=trg, attention_mask=trg_mask)['last_hidden_state']
        src, trg = self.layer_norm(src), self.layer_norm(trg)
        src, trg = src[torch.arange(batch_size), src_eos], trg[torch.arange(batch_size), trg_eos]

        src = F.normalize(self.src_wts(src))
        trg = F.normalize(self.trg_wts(trg))
        sim_output = torch.mm(src, trg.transpose(0, 1)) * self.temperature.exp()

        return sim_output, src, trg