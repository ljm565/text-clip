import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel



class BertClip(nn.Module):
    def __init__(self, config, tokenizer, device, ko=False):
        super(BertClip, self).__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.flag = config.flag
        assert self.flag in ['eos', 'avg', 'max']

        self.model = BertModel.from_pretrained('skt/kobert-base-v1') if ko \
            else BertModel.from_pretrained('bert-base-uncased')

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
    

    def get_features(self, src, trg):
        if self.flag == 'eos':
            src, trg = src[torch.arange(self.batch_size), self.src_eos], trg[torch.arange(self.batch_size), self.trg_eos]
        elif self.flag == 'avg':
            hidden_dim = src.size(-1)
            self.src_mask, self.trg_mask = self.src_mask.unsqueeze(-1).repeat(1, 1, hidden_dim), self.trg_mask.unsqueeze(-1).repeat(1, 1, hidden_dim)
            src, trg = src.masked_fill(self.src_mask == 0, 0), trg.masked_fill(self.trg_mask == 0, 0)
            src, trg = torch.mean(src, dim=1), torch.mean(trg, dim=1)
        elif self.flag == 'max':
            self.src_mask, self.trg_mask = self.src_mask.unsqueeze(-1).repeat(1, 1, hidden_dim), self.trg_mask.unsqueeze(-1).repeat(1, 1, hidden_dim)
            src, trg = src.masked_fill(self.src_mask == 0, 0), trg.masked_fill(self.trg_mask == 0, 0)
            src, trg = torch.amax(src, dim=1), torch.amax(trg, dim=1)
        return src, trg


    def forward(self, src, trg):
        self.batch_size = src.size(0)
        self.src_mask, self.trg_mask = self.make_mask(src), self.make_mask(trg)
        self.src_eos, self.trg_eos = self.find_eos(src), self.find_eos(trg)
        
        src, trg = self.model(input_ids=src, attention_mask=self.src_mask)['last_hidden_state'], self.model(input_ids=trg, attention_mask=self.trg_mask)['last_hidden_state']
        src, trg = self.layer_norm(src), self.layer_norm(trg)
        src, trg = self.get_features(src, trg)

        src = F.normalize(self.src_wts(src))
        trg = F.normalize(self.trg_wts(trg))
        cos_sim = torch.mm(src, trg.transpose(0, 1))
        sim_output = cos_sim * self.temperature.exp()

        return sim_output, src, trg, cos_sim