import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
import pickle
from tokenizer import Tokenizer
import random
import time
from tqdm import tqdm
from utils.config import Config
from utils.utils_func import *
from utils.utils_data import DLoader
from models.cede import CEDe
from models.bert import BERT
from transformers import top_k_top_p_filtering



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr
        self.max_len = self.config.max_len
        self.result_num = self.config.result_num

        # define tokenizer
        self.tokenizer = Tokenizer(self.config)
        self.config.vocab_size = self.tokenizer.vocab_size

        # # dataloader
        # if self.mode != 'test':
        torch.manual_seed(999)  # for reproducibility
        chatbot_data_path = os.path.join(*[self.base_path, 'data', 'processed', 'chatbot', 'all_data.pkl'])
        sentiment_data_path = os.path.join(*[self.base_path, 'data', 'processed', 'sentiment', 'all_data.pkl'])
        chatbot_data, sentiment_data = load_dataset(chatbot_data_path), load_dataset(sentiment_data_path)
        self.dataset = DLoader(chatbot_data, sentiment_data, self.tokenizer, self.config)
        data_size = len(self.dataset)
        train_size = int(data_size * 0.95)
        val_size = int(data_size * 0.03)
        test_size = data_size - train_size - val_size

        self.trainset, self.valset, self.testset = random_split(self.dataset, [train_size, val_size, test_size])
        if self.mode == 'train':
            self.dataset = {'train': self.trainset, 'val': self.valset, 'test': self.testset}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {'test': self.testset}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = BERT(self.config, self.tokenizer, self.device).to(self.device)
        self.chatbot_criterion = nn.CrossEntropyLoss()
        # self.sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
        if self.mode == 'train':
            total_steps = len(self.dataloaders['train']) * self.epochs
            pct_start = 100 / total_steps
            final_div_factor = self.lr / 25 / 1e-7    # OneCycleLR default value is 25
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
            print(self.lr)
            # self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            # milestones = list(range(8, self.epochs, 8))
            # self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.8)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                self.scheduler.load_state_dict(self.check_point['scheduler'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_loss = float('inf')
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val', 'test']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    epoch_loss = self.train(phase, epoch)
                    train_loss_history.append(epoch_loss)
                    # self.scheduler.step()
                else:
                    loss = self.inference(phase)
                    if phase == 'val':
                        val_loss_history.append(loss)

                        # save best model
                        early_stop += 1
                        if  loss < best_val_loss:
                            early_stop = 0
                            best_val_loss = loss
                            best_epoch = best_epoch_info + epoch + 1
                            save_checkpoint(self.model_path, self.model, self.optimizer)
                            
                            self.loss_data = {'best_epoch': best_epoch, 'best_val_sloss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
                            print('Saving the loss related data...')
                            with open(self.config.loss_data_path, 'wb') as f:
                                pickle.dump(self.loss_data, f)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))
        self.loss_data = {'best_epoch': best_epoch, 'best_val_sloss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
        return self.loss_data


    def train(self, phase, epoch):
        self.model.train()
        total_loss = 0

        for i, (src, trg) in enumerate(self.dataloaders[phase]):
            batch_size = src.size(0)
            self.optimizer.zero_grad()
            src, trg = src.to(self.device), trg.to(self.device)

            with torch.set_grad_enabled(phase=='train'):
                sim_output, _, _ = self.model(src, trg)

                label = torch.arange(batch_size).to(self.device)
                loss = (self.chatbot_criterion(sim_output, label) + self.chatbot_criterion(sim_output.transpose(0, 1), label)) / 2
                loss.backward()
                self.optimizer.step()

            total_loss +=  loss.item() * batch_size

            if i % 100 == 0:
                print('Epoch {}: {}/{} step loss: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item()))

        epoch_loss = total_loss / len(self.dataloaders[phase].dataset)

        print('{} loss: {} \n'.format(phase, epoch_loss))
        return epoch_loss


    def inference(self, phase):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                batch_size = src.size(0)
                src, trg = src.to(self.device), trg.to(self.device)
                sim_output, _, _ = self.model(src, trg)
                
                label = torch.arange(batch_size).to(self.device)
                loss = (self.chatbot_criterion(sim_output, label) + self.chatbot_criterion(sim_output.transpose(0, 1), label)) / 2

                total_loss += loss.item() * batch_size


        print('\nInference Result')
        print('loss: {}'.format(total_loss/len(self.dataloaders[phase].dataset)))

        return total_loss/len(self.dataloaders[phase].dataset)


    def test(self, phase):
        topk = 10
        self.model.eval()
        all_txt, all_trg_txt, all_src_emb, all_trg_emb = [], [], [], []

        with torch.no_grad():
            for src, trg in tqdm(self.dataloaders[phase], desc=phase + ' inferencing..'):
                all_txt += tensor2list(src, self.tokenizer)
                all_trg_txt += tensor2list(trg, self.tokenizer)
                
                src, trg = src.to(self.device), trg.to(self.device)
                _, src_emb, trg_emb = self.model(src, trg)
                
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
            
            _, emb, _, = self.model(s, s)
            sim = torch.mm(emb.detach().cpu(), all_trg_emb.transpose(0, 1))
            score, idx = torch.sort(sim.squeeze(), descending=True)
            score, idx = score[:topk], idx[:topk]
            
            print('src: ', sen)
            print('gt    ({:4f}): {}'.format(sim[0, gt_id], gt_trg))
            for n, (i, s) in enumerate(zip(idx, score)):
                print('trg {} ({:4f}): {}'.format(n, s, all_trg_txt[i]))
            print('-'*50 + '\n\n')





        top1_p, top5_p, top10_p = 0, 0, 0

        # for txt_id in tqdm(range(len(all_txt))):
        #     sen = all_txt[txt_id]
        #     s = [self.tokenizer.cls_token_id] + self.tokenizer.encode(sen)[:self.max_len-2] + [self.tokenizer.sep_token_id]
        #     s = s + [self.tokenizer.pad_token_id] * (self.max_len - len(s))
        #     s = torch.LongTensor(s).to(self.device).unsqueeze(0)
            
        #     _, emb, _, = self.model(s, s)
        #     sim = torch.mm(emb.detach().cpu(), all_trg_emb.transpose(0, 1))# * self.model.temperature.exp()
        #     _, idx = torch.sort(sim.squeeze(), descending=True)
        #     idx = idx[:topk]

        #     top1_p += isInTopk(txt_id, idx, 1)
        #     top5_p += isInTopk(txt_id, idx, 5)
        #     top10_p += isInTopk(txt_id, idx, 10)
        
        # print('top1: {}'.format(top1_p / len(all_txt)))
        # print('top5: {}'.format(top5_p / len(all_txt)))
        # print('top10: {}'.format(top10_p / len(all_txt)))