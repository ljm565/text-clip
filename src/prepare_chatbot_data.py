import os
import pickle
import pandas as pd
from tqdm import tqdm
from utils.utils_prepare_chatbot import *



def prepare_data_type1(base_path, folder_name):
    folder_path = os.path.join(base_path, folder_name)
    files = os.listdir(folder_path)
    
    for file in tqdm(files, desc=folder_name + ' processing...'):
        df = pd.read_excel(os.path.join(folder_path, file), engine='openpyxl')
        speaker_ids, sen_ids, sentences = df['SPEAKERID'].tolist(), df['SENTENCEID'].tolist(), df['SENTENCE'].tolist()

        # add '.' and merge same speaker
        sentences = add_punctuation(sentences)
        speaker_ids, sen_ids, sentences = type1_merge_same_speaker(speaker_ids, sen_ids, sentences)

        # cut not multi-turn data
        if speaker_ids[-1] == 1:
            speaker_ids, sen_ids, sentences = speaker_ids[:-1], sen_ids[:-1], sentences[:-1]
        
        # merge multi-turn data
        multiturns = type1_merge_multiturn(sen_ids, sentences)

        # save multi-turn data
        save_path = [base_path[:-12], 'processed', 'chatbot', folder_name + '_' + file[:-5] + '.pkl']
        save_path = os.path.join(*save_path)
        
        with open(save_path, 'wb') as f:
            pickle.dump(multiturns, f)



def prepare_data_type2(base_path, folder_name):
    folder_path = os.path.join(base_path, folder_name)
    files = os.listdir(folder_path)
    
    for file in tqdm(files, desc=folder_name + ' processing...'):
        df = pd.read_csv(os.path.join(folder_path, file), low_memory=False)
        sen_ids, sentences = df['상담번호'].tolist(), df['발화문'].tolist()
        
        multiturn_dict = {}
        for sen_id, sentence in zip(sen_ids, sentences):
            if sen_id in multiturn_dict.keys():
                multiturn_dict[sen_id].append(add_punctuation(sentence, False))
            else:
                multiturn_dict[sen_id] = [sentence]

        # save multi-turn data
        multiturns = [tuple(v) for v in multiturn_dict.values()]
        save_path = [base_path[:-12], 'processed', 'chatbot',  folder_name + '_' + file[:-4] + '.pkl']
        save_path = os.path.join(*save_path)
        
        with open(save_path, 'wb') as f:
            pickle.dump(multiturns, f)
 


def prepare_data_type3(base_path, folder_name):
    folder_path = os.path.join(base_path, folder_name)

    files = []
    for (root, _, tmp) in os.walk(folder_path):
        for file in tmp:
            files.append(os.path.join(root, file))
    
    multiturns = []
    for file in tqdm(files, desc=folder_name + ' processing...'):
        with open(file, 'r') as f:
            multiturn = f.readlines()
        multiturn = [type3_add_punctuation(l.strip()) for l in multiturn]
        multiturn = type3_merge_same_speaker(multiturn)
        multiturns.append(tuple(multiturn))

    # save multi-turn data
    save_path = [base_path[:-12], 'processed', 'chatbot',  folder_name + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(multiturns, f)



def prepare_data_type4(base_path, folder_name):
    folder_path = os.path.join(base_path, folder_name)

    files = []
    for (root, _, tmp) in os.walk(folder_path):
        for file in tmp:
            files.append(os.path.join(root, file))
    
    multiturns = []
    for file in tqdm(files, desc=folder_name + ' processing...'):
        with open(file, 'r') as f:
            multiturn = f.readlines()
        multiturn = [l.strip() for l in multiturn]
        multiturn = type4_merge_same_speaker(multiturn, file)
        multiturns.append(tuple(multiturn))

    # save multi-turn data
    save_path = [base_path[:-12], 'processed', 'chatbot',  folder_name + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(multiturns, f)