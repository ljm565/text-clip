import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import time
import pickle
import random
from tqdm import tqdm
from scipy import stats
# from sentence_transformers import SentenceTransformer

from models.bert import BertClip
from utils.utils_func import *
from tokenizer import Tokenizer
from utils.config import Config
from utils.utils_data import DLoader, SemanticDLoader
import losses





class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int, isSBERT:bool=False):
        torch.manual_seed(999)  # for reproducibility

        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.isSBERT = isSBERT
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_name = self.config.data
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.max_len = self.config.max_len
        self.result_num = self.config.result_num
        self.train_mode = self.config.train_mode

        # define tokenizer
        self.tokenizer = Tokenizer()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # dataloader
        if not self.mode == 'benchmark':
            self.dataset = {split + '_' + m: SemanticDLoader(load_dataset(p), self.tokenizer, self.config) \
                for split, mode in self.config.dataset_path.items() for m, p in mode.items()}
        else:
            self.dataset = {s: SemanticDLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.config.dataset_path.items()}

        if self.mode == 'train':
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if 'train' in s else DataLoader(d, self.batch_size, shuffle=False)
                    for s, d in self.dataset.items()}
        else:
            tmp = 'test' if not 'semantic' in self.data_name else 'val'
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if tmp in s}

        # model, optimizer, losses
        self.model = BertClip(self.config, self.tokenizer, self.device).to(self.device)
        self.nli_loss = losses.SoftmaxLoss(self.model.hidden_dim, 3).to(self.device)
        self.clip_loss = losses.ClipLoss().to(self.device)
        if self.mode == 'train':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            if self.isSBERT:
                self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
            else:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])    
                self.model.eval()
                del self.check_point
                torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_loss = float('inf')

        steps = ['train', 'val']
        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in steps:
                for task in self.train_mode:
                    print('Phase/mode: {}/{}'.format(phase, task))
                    if phase == 'train':
                        epoch_loss = 0
                        if task == 'nli':
                            nli_loss = self.nli_train(phase, epoch)
                            epoch_loss += nli_loss
                        if task == 'clip':
                            clip_loss = self.clip_train(phase, epoch)
                            epoch_loss += clip_loss
                        # try:
                        #     epoch_loss = self.clip_train(phase, epoch)
                        # except KeyError:
                        #     epoch_loss = 0
                        #     for mode in self.train_mode:
                        #         if mode == 'clip':
                        #             loss1 = self.clip_train(phase + '_clip', epoch)
                        #             epoch_loss += loss1
                        #         elif mode == 'nli':
                        #             loss2 = self.nli_train(phase + '_nli', epoch)
                        #             epoch_loss += loss2
                        #         elif mode == 'reg':
                        #             loss3 = self.reg_train(phase + '_reg', epoch)
                        #             epoch_loss += loss3
                        #     # epoch_loss = loss1 + loss2 + loss3
                    else:
                        epoch_loss = 0
                        if task == 'nli':
                            nli_loss = self.nli_inference(phase, epoch)
                            epoch_loss += nli_loss
                        if task == 'clip':
                            clip_loss = self.clip_inference(phase, epoch)
                            epoch_loss += clip_loss
                        # try:
                        #     epoch_loss = self.clip_inference(phase)
                        # except KeyError:
                        #     epoch_loss = 0
                        #     for mode in self.train_mode:
                        #         if mode == 'clip':
                        #             loss1 = self.clip_inference(phase + '_clip')
                        #             epoch_loss += loss1
                        #         elif mode == 'nli':
                        #             loss2 = self.nli_inference(phase + '_nli')
                        #             epoch_loss += loss2
                        #         elif mode == 'reg':
                        #             loss3 = self.reg_inference(phase + '_reg')
                        #             epoch_loss += loss3
                        #     # loss = loss1 + loss2 + loss3

                        if phase == 'val':
                            # save best model
                            early_stop += 1
                            if  epoch_loss < best_val_loss:
                                early_stop = 0
                                best_val_loss = epoch_loss
                                best_epoch = epoch + 1
                                save_checkpoint(self.model_path, self.model, self.optimizer)
                            
            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))


    def clip_train(self, phase, epoch):
        print('clip training starts')
        self.model.train()
        total_loss = 0
        for i, (src, trg, _) in enumerate(self.dataloaders[phase+'_clip']):
            batch_size = src.size(0)
            self.optimizer.zero_grad()
            src, trg = src.to(self.device), trg.to(self.device)

            with torch.set_grad_enabled('train' in phase):
                src, trg = self.model(src, trg)
                label = torch.arange(batch_size).to(self.device)
                loss = self.clip_loss([src, trg], label)
                loss.backward()
                self.optimizer.step()

            total_loss +=  loss.item() * batch_size

            if i % 1000 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
            
        epoch_loss = total_loss / len(self.dataloaders[phase].dataset)

        print('{} loss: {} \n'.format(phase, epoch_loss))
        return epoch_loss

    
    def nli_train(self, phase, epoch):
        print('nli training starts')
        self.model.train()
        total_loss = 0
        phase = phase+'_nli'
        for i, (src, trg, label) in enumerate(self.dataloaders[phase]):
            batch_size = src.size(0)
            self.optimizer.zero_grad()
            src, trg, label = src.to(self.device), trg.to(self.device), label.to(self.device)

            with torch.set_grad_enabled('train' in phase):
                src, trg = self.model(src, trg)
                loss = self.nli_loss([src, trg], label)
                loss.backward()
                self.optimizer.step()

            total_loss +=  loss.item() * batch_size

            if i % 1000 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
            
        epoch_loss = total_loss / len(self.dataloaders[phase].dataset)

        print('{} loss: {} \n'.format(phase, epoch_loss))
        return epoch_loss


    def reg_train(self, phase, epoch):
        print('regression training starts')
        self.model.train()
        total_loss = 0
        for i, (src, trg, label) in enumerate(self.dataloaders[phase]):
            batch_size = src.size(0)
            self.optimizer.zero_grad()
            src, trg, label = src.to(self.device), trg.to(self.device), label.to(self.device)
            
            with torch.set_grad_enabled('train' in phase):
                _, _, _, cos_sim, _ = self.model(src, trg)

                loss = self.reg_criterion(cos_sim[torch.arange(batch_size), torch.arange(batch_size)], label)
                loss.backward()
                self.optimizer.step()

            total_loss +=  loss.item() * batch_size
            
            if i % 1000 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))
            
        epoch_loss = total_loss / len(self.dataloaders[phase].dataset)

        print('{} loss: {} \n'.format(phase, epoch_loss))
        return epoch_loss


    def clip_inference(self, phase):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for src, trg, _ in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                batch_size = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)
                src, trg = self.model(src, trg)
                label = torch.arange(batch_size).to(self.device)
                loss = self.clip_loss([src, trg], label)
                total_loss += loss.item() * batch_size
            
            print('loss: {}\n'.format(total_loss/len(self.dataloaders[phase].dataset)))
            return total_loss/len(self.dataloaders[phase].dataset)

    
    def nli_inference(self, phase):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for src, trg, label in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                batch_size = src.size(0)
                src, trg, label = src.to(self.device), trg.to(self.device), label.to(self.device)
                src, trg = self.model(src, trg)
                loss = self.nli_loss([src, trg], label)
                total_loss += loss.item() * batch_size
            print('loss: {}\n'.format(total_loss/len(self.dataloaders[phase].dataset)))
            return total_loss/len(self.dataloaders[phase].dataset)


    def reg_inference(self, phase):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for src, trg, label in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                batch_size = src.size(0)
                src, trg, label = src.to(self.device), trg.to(self.device), label.to(self.device)
                _, _, _, cos_sim, _ = self.model(src, trg)
                
                loss = self.reg_criterion(cos_sim[torch.arange(batch_size), torch.arange(batch_size)], label)

                total_loss += loss.item() * batch_size
            
            print('loss: {}'.format(total_loss/len(self.dataloaders[phase].dataset)))
            print()
            return total_loss/len(self.dataloaders[phase].dataset)


    def test(self, phase):
        topk = 10
        self.model.eval()
        all_txt, all_trg_txt, all_src_emb, all_trg_emb = [], [], [], []

        with torch.no_grad():
            for src, trg, _ in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                all_txt += tensor2list(src, self.tokenizer)
                all_trg_txt += tensor2list(trg, self.tokenizer)
                
                src, trg = src.to(self.device), trg.to(self.device)
                _, src_emb, trg_emb, _ = self.model(src, trg)
                
                all_trg_emb.append(trg_emb.detach().cpu())
                all_src_emb.append(src_emb.detach().cpu())

        all_trg_emb = torch.cat(all_trg_emb, dim=0)
        all_src_emb = torch.cat(all_src_emb, dim=0)
        
        print('\n\n')
        for _ in range(100):
            gt_id = random.randrange(len(all_txt))
            sen = all_txt[gt_id]
            gt_trg = all_trg_txt[gt_id]

            s = [self.tokenizer.cls_token_id] + self.tokenizer.encode(sen)[:self.max_len-2] + [self.tokenizer.sep_token_id]
            s = s + [self.tokenizer.pad_token_id] * (self.max_len - len(s))
            s = torch.LongTensor(s).to(self.device).unsqueeze(0)
            
            _, emb, _, _ = self.model(s, s)
            sim = torch.mm(emb.detach().cpu(), all_trg_emb.transpose(0, 1))
            score, idx = torch.sort(sim.squeeze(), descending=True)
            score, idx = score[:topk], idx[:topk]
            
            print('src: ', sen)
            print('gt    ({:4f}): {}'.format(sim[0, gt_id], gt_trg))
            for n, (i, s) in enumerate(zip(idx, score)):
                print('trg {} ({:4f}): {}'.format(n, s, all_trg_txt[i]))
            print('-'*50 + '\n\n')

        top1_p, top5_p, top10_p = 0, 0, 0
        for txt_id in tqdm(range(len(all_txt))):
            sen = all_txt[txt_id]
            s = [self.tokenizer.cls_token_id] + self.tokenizer.encode(sen)[:self.max_len-2] + [self.tokenizer.sep_token_id]
            s = s + [self.tokenizer.pad_token_id] * (self.max_len - len(s))
            s = torch.LongTensor(s).to(self.device).unsqueeze(0)
            
            _, emb, _, _ = self.model(s, s)
            sim = torch.mm(emb.detach().cpu(), all_trg_emb.transpose(0, 1))# * self.model.temperature.exp()
            _, idx = torch.sort(sim.squeeze(), descending=True)
            idx = idx[:topk]

            top1_p += isInTopk(txt_id, idx, 1)
            top5_p += isInTopk(txt_id, idx, 5)
            top10_p += isInTopk(txt_id, idx, 10)
        
        print('top1: {}'.format(top1_p / len(all_txt)))
        print('top5: {}'.format(top5_p / len(all_txt)))
        print('top10: {}'.format(top10_p / len(all_txt)))



    def benchmark_test(self, phase):
        self.model.eval()
        all_cosSim, all_l = [], []

        with torch.no_grad():
            for src, trg, l in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                if self.isSBERT:
                    src, trg = src.to(self.device), trg.to(self.device)
                    src, trg = [self.tokenizer.decode(s.tolist()) for s in src], [self.tokenizer.decode(s.tolist()) for s in trg]
                    src_emb, trg_emb = torch.from_numpy(self.model.encode(src)), torch.from_numpy(self.model.encode(trg))
                    cos_sim = torch.diagonal(torch.mm(src_emb, trg_emb.transpose(0, 1)))
                else:
                    src, trg = src.to(self.device), trg.to(self.device)
                    _, _, _, cos_sim, _ = self.model(src, trg)
                    cos_sim = torch.diagonal(cos_sim)
                # l = (l / 5) * 2 - 1
                
                all_cosSim.append(cos_sim.detach().cpu())
                all_l.append(l.detach().cpu())

        all_cos = torch.cat(all_cosSim, dim=0).numpy()
        all_l = torch.cat(all_l, dim=0).numpy()

        res = stats.spearmanr(all_cos, all_l)
        print(res.statistic)
        print(all_cos[:10])
        print(all_l[:10])