import random
import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, chat_data, tokenizer, config):
        random.seed(999) 
        self.data = self.construct_biturn_data(list(filter(lambda x: len(x) >= 2, chat_data)))
        random.shuffle(self.data)
        self.tokenizer = tokenizer
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_len = config.max_len
        self.length = len(self.data)


    def tokenize_(self, data):
        src = [self.cls_token_id] + self.tokenizer.encode(data[0])[:self.max_len-2] + [self.sep_token_id]
        trg = [self.cls_token_id] + self.tokenizer.encode(data[1])[:self.max_len-2] + [self.sep_token_id]
        
        src = src + [self.pad_token_id] * (self.max_len - len(src))
        trg = trg + [self.pad_token_id] * (self.max_len - len(trg))
        return src, trg


    def construct_biturn_data(self, data):
        biturn_data = []
        for d in data:
            for s1, s2 in zip(d[:-1], d[1:]):
                biturn_data.append((s1, s2))
        return biturn_data


    def __getitem__(self, idx):
        d = self.data[idx]
        s, t = self.tokenize_(d)

        return torch.LongTensor(s), torch.LongTensor(t)


    def __len__(self):
        return self.length