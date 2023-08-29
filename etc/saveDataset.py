import os
import pickle
from datasets import load_dataset



def init_data():
    trainset, valset, testset = {}, {}, {}

    # mrpc
    trainset['mrpc'] = load_dataset('glue', 'mrpc', split='train')
    valset['mrpc']   = load_dataset('glue', 'mrpc', split='validation')
    testset['mrpc']  = load_dataset('glue', 'mrpc', split='test')

    # snli
    trainset['snli'] = load_dataset('snli', split='train')
    valset['snli']   = load_dataset('snli', split='validation')
    testset['snli']  = load_dataset('snli', split='test')

    # qqp
    trainset['qqp'] = load_dataset('glue', 'qqp', split='train')
    valset['qqp']   = load_dataset('glue', 'qqp', split='validation')
    testset['qqp']  = load_dataset('glue', 'qqp', split='test')

    # sts-b
    trainset['stsb'] = load_dataset('glue', 'stsb', split='train')
    valset['stsb']   = load_dataset('glue', 'stsb', split='validation')
    testset['stsb']  = load_dataset('glue', 'stsb', split='test')

    # mnli
    trainset['mnli'] = load_dataset('glue', 'mnli', split='train')
    valset['mnli']   = [load_dataset('glue', 'mnli', split='validation_matched'), load_dataset('glue', 'mnli', split='validation_mismatched')]
    testset['mnli']  = [load_dataset('glue', 'mnli', split='test_matched'), load_dataset('glue', 'mnli', split='test_mismatched')]

    return trainset, valset, testset



def save_data(phase, dataset, mode, path):
    all_pairs = []

    if mode == 'clip':
        save_clip_data(phase, dataset, path)
    elif mode == 'nli':
        save_nli_data(phase, dataset, path)
    elif mode == 'reg':
        save_reg_data(phase, dataset, path)



def save_clip_data(phase, dataset, path):
    all_pairs = []
    save_p = os.path.join(*[path, phase, 'clip.' + phase])
    os.makedirs(os.path.join(path, phase), exist_ok=True)

    for d in dataset['mrpc']:
        if d['label'] == 1:
            s1, s2 = d['sentence1'], d['sentence2']
            all_pairs.append((s1, s2, 1))

    print(f'after mrpc, {len(all_pairs)} pairs')


    for d in dataset['snli']:
        if d['label'] == 0:
            s1, s2 = d['premise'], d['hypothesis']
            all_pairs.append((s1, s2, 1))

    print(f'after snli, {len(all_pairs)} pairs')


    for d in dataset['qqp']:
        if d['label'] == 1:
            s1, s2 = d['question1'], d['question2']
            all_pairs.append((s1, s2, 1))

    print(f'after qqp, {len(all_pairs)} pairs')


    for d in dataset['stsb']:
        if d['label'] >= 3:
            s1, s2 = d['sentence1'], d['sentence2']
            all_pairs.append((s1, s2, 1))

    print(f'after stsb, {len(all_pairs)} pairs')


    if phase == 'train':
        for d in dataset['mnli']:
            if d['label'] == 0:
                s1, s2 = d['premise'], d['hypothesis']
                all_pairs.append((s1, s2, 1))
    else:
        for sets in dataset['mnli']:
            for d in sets:
                if d['label'] == 0:
                    s1, s2 = d['premise'], d['hypothesis']
                    all_pairs.append((s1, s2, 1))

    print(f'after mnli, {len(all_pairs)} pairs')
    print()

    with open(save_p, 'wb') as f:
        pickle.dump(all_pairs, f)




def save_nli_data(phase, dataset, path):
    all_pairs = []
    save_p = os.path.join(*[path, phase, 'nli.' + phase])
    os.makedirs(os.path.join(path, phase), exist_ok=True)


    for d in dataset['snli']:
        if d['label'] == -1:
            continue
        s1, s2, label = d['premise'], d['hypothesis'], d['label']
        all_pairs.append((s1, s2, label))

    print(f'after snli, {len(all_pairs)} pairs')


    if phase == 'train':
        for d in dataset['mnli']:
            s1, s2, label = d['premise'], d['hypothesis'], d['label']
            all_pairs.append((s1, s2, label))
    else:
        for sets in dataset['mnli']:
            for d in sets:
                s1, s2, label = d['premise'], d['hypothesis'], d['label']
                all_pairs.append((s1, s2, label))

    print(f'after mnli, {len(all_pairs)} pairs')
    print()

    with open(save_p, 'wb') as f:
        pickle.dump(all_pairs, f)




def save_reg_data(phase, dataset, path):
    all_pairs = []
    save_p = os.path.join(*[path, phase, 'reg.' + phase])
    os.makedirs(os.path.join(path, phase), exist_ok=True)


    for d in dataset['mrpc']:
        if d['label'] == 1:
            s1, s2 = d['sentence1'], d['sentence2']
            all_pairs.append((s1, s2, 1))

        elif d['label'] == 0:
            s1, s2 = d['sentence1'], d['sentence2']
            all_pairs.append((s1, s2, 0))

    print(f'after mrpc, {len(all_pairs)} pairs')


    for d in dataset['qqp']:
        if d['label'] == 1:
            s1, s2 = d['question1'], d['question2']
            all_pairs.append((s1, s2, 1))
        elif d['label'] == 0:
            s1, s2 = d['question1'], d['question2']
            all_pairs.append((s1, s2, 0))

    print(f'after qqp, {len(all_pairs)} pairs')


    for d in dataset['stsb']:
        s1, s2, label = d['sentence1'], d['sentence2'], d['label']
        all_pairs.append((s1, s2, label/5))

    print(f'after stsb, {len(all_pairs)} pairs')

    # for d in dataset['snli']:
    #     if d['label'] == 1:
    #         s1, s2 = d['premise'], d['hypothesis']
    #         all_pairs.append((s1, s2, 0))
    #     elif d['label'] == 2:
    #         s1, s2 = d['premise'], d['hypothesis']
    #         all_pairs.append((s1, s2, -1))

    # print(f'after snli, {len(all_pairs)} pairs')

    # if phase == 'train':
    #     for d in dataset['mnli']:
    #         if d['label'] == 1:
    #             s1, s2 = d['premise'], d['hypothesis']
    #             all_pairs.append((s1, s2, 0))
    #         elif d['label'] == 2:
    #             s1, s2 = d['premise'], d['hypothesis']
    #             all_pairs.append((s1, s2, -1))
    # else:
    #     for sets in dataset['mnli']:
    #         for d in sets:
    #             if d['label'] == 1:
    #                 s1, s2 = d['premise'], d['hypothesis']
    #                 all_pairs.append((s1, s2, 0))
    #             elif d['label'] == 1:
    #                 s1, s2 = d['premise'], d['hypothesis']
    #                 all_pairs.append((s1, s2, -1))

    print(f'after mnli, {len(all_pairs)} pairs')
    print()


    with open(save_p, 'wb') as f:
        pickle.dump(all_pairs, f)



if __name__ == '__main__':
    trainset, valset, testset = init_data()
    
    path = './data/'
    for mode in ['clip', 'nli', 'reg']:
        for phase in ['train', 'val']:
            if phase == 'train':
                save_data(phase, trainset, mode, path)
            else:
                save_data(phase, valset, mode, path)