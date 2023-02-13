import os
import pickle
import pandas as pd
from tqdm import tqdm
from utils.utils_prepare_sentiment import *



def prepare_sdata_type1(base_path, file_name):
    with open(base_path + file_name, 'r') as f:
        lines = f.readlines()
    
    lines = [s.strip().split('\t') for s in lines]
    lines = type1_reivew_matcher(lines, file_name)

    # save multi-turn data
    save_path = [base_path[:-15], 'processed', 'sentiment', file_name[:-4] + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(lines, f)



def prepare_sdata_type2(base_path, file_name):
    with open(base_path + file_name, 'r') as f:
        lines = f.readlines()
    
    lines = [s.strip().split('\t') for s in lines]
    lines = type2_reivew_matcher(lines, file_name)

    # save multi-turn data
    save_path = [base_path[:-15], 'processed', 'sentiment', file_name[:-4] + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(lines, f)



def prepare_sdata_type3(base_path, file_name):
    with open(base_path + file_name, 'r') as f:
        lines = f.readlines()
    
    lines = [s.strip().split('\t') for s in lines]
    lines = type3_reivew_matcher(lines, file_name)

    # save multi-turn data
    save_path = [base_path[:-15], 'processed', 'sentiment', file_name[:-4] + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(lines, f)



def prepare_sdata_type4(base_path, file_name):
    df = pd.read_csv(os.path.join(base_path, file_name))
    reviews, scores = df['발화'].tolist(), df['감정_int'].tolist()

    all_reviews = []
    for review, score in zip(tqdm(reviews, desc=file_name + ' processing...'), scores):
        all_reviews.append((score, review.strip()))

    # save multi-turn data
    save_path = [base_path[:-15], 'processed', 'sentiment', file_name[:-4] + '.pkl']
    save_path = os.path.join(*save_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(all_reviews, f)