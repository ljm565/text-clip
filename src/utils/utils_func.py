import torch
import re
import os
import pickle
import random
from tqdm import tqdm
import pandas as pd



"""
common utils
"""
def collect_all_data(path):
    random.seed(999)

    files = os.listdir(path)
    all_data = []
    for file in files:
        with open(path + file, 'rb') as f:
            data = pickle.load(f)
        all_data.append(data)
    all_data = sum(all_data, [])
    
    random.shuffle(all_data)
    
    with open(path + 'all_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    
    return all_data


def make_vocab_file(data, data_save_path, special_token_save_path):
    all_s = []
    special_tok = set()
    for d in tqdm(data, desc='making vocab train file...'):
        for s in d:
            # find special token e.g. #@이름#
            compile = re.compile(r'(?<=#@)(.*?)(?=#)')
            special_tok.update(compile.findall(s))
            s = ' '.join(s.strip().split())
            all_s.append(s + '\n')
    
    with open(data_save_path, 'w') as f:
        f.writelines(all_s)
    
    all_special_tokens = []
    for tok in list(special_tok):
        all_special_tokens.append('#@' + tok + '#\n')

    with open(special_token_save_path, 'w') as f:
        f.writelines(all_special_tokens)



def load_dataset(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except:
        data = pd.read_csv(path, delimiter='\t', header=None, keep_default_na=False)
        try:
            s1, s2, s3 = data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist(), data.iloc[:, 2].tolist()
            data = [(ss1, ss2, ss3) for ss1, ss2, ss3 in zip(s1, s2, s3)]
        except IndexError:
            s1, s2 = data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()
            data = [(ss1, ss2) for ss1, ss2 in zip(s1, s2)]
    return data


def save_checkpoint(file, model, optimizer, scheduler=None):
    if scheduler == None:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    else:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def tensor2list(tensor, tokenizer):
    tensor = tensor.detach().cpu().tolist()
    tensor = [tokenizer.decode(t[1:]) for t in tensor]
    return tensor


def isInTopk(gt_id, selected_id, topk):
    selected_id = selected_id.tolist()
    if len(selected_id) < topk:
        raise IndexError("topk is larger than selected id")

    if gt_id in selected_id[:topk]:
        return 1
    return 0


def make_dataset_path(base_path, data_name, train_mode=None):
    dataset_path = {}
    d_path = base_path + 'data/' + data_name + '/processed/'
    phase = list(set([file[file.rfind('.')+1:] for file in os.listdir(d_path)]))

    for split in phase:
        if train_mode == None:
            dataset_path[split] = d_path + data_name + '.' + split
        else:
            dataset_path[split] = {}
            for mode in train_mode:
                dataset_path[split][mode] = d_path + 'semantic' + '_' + mode + '.' + split
    
    return dataset_path


# def switch_tensor(x, mask):
#     batch_size = x.size(0)
    
#     mask = mask.unsqueeze(1)
#     mask = mask.repeat(1, batch_size)
#     mask = torch.where(mask == 0, -1, 1)

#     return x*mask