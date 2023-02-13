import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# word embedding layer
class TokEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(TokEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional embedding layer
class PosEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, device):
        super(PosEmbedding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.device = device

        self.pos = torch.arange(0, self.max_len)
        self.emb_layer = nn.Embedding(self.max_len, self.hidden_dim)


    def forward(self, x):
        return self.emb_layer(self.pos.unsqueeze(0).to(self.device))[:, :x.size(1)]
        



# mulithead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, self_attn, causal):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.self_attn = self_attn
        self.causal = causal
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim == self.num_head * self.head_dim
        
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=self.bias)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)


    def head_split(self, x):
        x = x.view(self.batch_size, -1, self.num_head, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x


    def scaled_dot_product(self, q, k, v, mask):
        attn_wts = torch.matmul(q, torch.transpose(k, 2, 3))/(self.head_dim ** 0.5)
        if not mask == None:
            attn_wts = attn_wts.masked_fill(mask==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_out = torch.matmul(attn_wts, v)
        return attn_wts, attn_out


    def reshaping(self, attn_out):
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(self.batch_size, -1, self.hidden_dim)
        return attn_out


    def forward(self, x, mask):
        self.batch_size = x.size(0)
        q, k, v = self.qkv(x).split(self.hidden_dim, dim=2)

        q = self.head_split(q)
        k = self.head_split(k)
        v = self.head_split(v)

        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# postion wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout, bias):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.bias = bias

        self.FFN1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=self.bias),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=self.bias),
        )
        self.init_weights()


    def init_weights(self):
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.02)


    def forward(self, x):
        output = self.FFN1(x)
        output = self.FFN2(output)
        return output



# a single decoder layer
class SingleBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(SingleBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.masked_self_atention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=True)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, mask):
        x = self.layer_norm(x)
        dec_self_attn_wts, output = self.masked_self_atention(x, mask=mask)
        output = self.dropout_layer(output)
        output = x + output

        x = output
        output = self.positionWiseFeedForward(self.layer_norm(output))
        output = self.dropout_layer(output)
        output = x + output

        return dec_self_attn_wts, output



# all decoders
class Blocks(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Blocks, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.device = device

        self.dec_num_layers = config.dec_num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps

        self.dropout_layer = nn.Dropout(self.dropout)
        self.tok_layer = TokEmbedding(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_layer = PosEmbedding(self.max_len*2, self.hidden_dim, self.device)
        self.decoders = nn.ModuleList([SingleBlock(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.dec_num_layers)])


    def forward(self, src, mask):
        output = self.tok_layer(src) + self.pos_layer(src)
        output = self.dropout_layer(output)

        all_self_attn_wts = []
        for decoder in self.decoders:
            dec_self_attn_wts, output = decoder(output, mask)
            all_self_attn_wts.append(dec_self_attn_wts.detach().cpu())
        
        return all_self_attn_wts, output



# CEDe
class CEDe(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(CEDe, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.layernorm_eps = config.layernorm_eps
        self.bias = config.bias
        
        self.hidden_dim = self.config.hidden_dim
        self.max_len = config.max_len

        self.decoder1 = Blocks(self.config, self.tokenizer, self.device)
        self.decoder2 = Blocks(self.config, self.tokenizer, self.device)


        self.src_wts = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.trg_wts = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sentiment_fc = nn.Linear(self.hidden_dim, 2, bias=self.bias)




    def make_data_id(self, sentiment_cls):
        chatbot_id = sentiment_cls == -1
        sentiment_id = sentiment_cls != -1
        return chatbot_id, sentiment_id


    def make_mask(self, src):
        mask = torch.where(src==self.tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        return mask
        

    def forward(self, src, trg, sentiment_cls):
        chatbot_id, sentiment_id = self.make_data_id(sentiment_cls)
        chatbot_src, chatbot_trg = src[chatbot_id].to(self.device), trg[chatbot_id].to(self.device)
        sentiment_src = src[sentiment_id].to(self.device)
        chatbot_src_mask, chatbot_trg_mask, sentiment_mask = self.make_mask(chatbot_src), self.make_mask(chatbot_trg), self.make_mask(sentiment_src)

        # chatbot output
        _, chatbot_src = self.decoder1(chatbot_src, chatbot_src_mask)
        _, chatbot_trg = self.decoder2(chatbot_trg, chatbot_trg_mask)

        chatbot_src = F.normalize(self.src_wts(chatbot_src[:, 0]))
        chatbot_trg = F.normalize(self.trg_wts(chatbot_trg[:, 0]))
        chatbot_output = torch.mm(chatbot_src, chatbot_trg.transpose(0, 1)) * self.temperature.exp()

        # sentiment output
        sentiment_output, sentiment_size = None, 0 
        if torch.sum(sentiment_id):
            _, sentiment_output = self.decoder1(sentiment_src, sentiment_mask)
            sentiment_output = self.sentiment_fc(sentiment_output)[:, 0]
            sentiment_size = sentiment_output.size(0)
        return (chatbot_output, chatbot_id, chatbot_output.size(0)), \
                (sentiment_output, sentiment_id, sentiment_size), (chatbot_src, chatbot_trg)