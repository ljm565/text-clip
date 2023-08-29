import torch
import pickle
import os
import json
import sys
from argparse import ArgumentParser
from train import Trainer
from utils.config import Config
from utils.utils_func import collect_all_data, make_vocab_file, make_dataset_path



def main(config_path:Config, args:ArgumentParser):
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif 'gpu' in args.device:
        device = torch.device('cuda:0') if args.device == 'gpu' else torch.device(f'cuda:{args.device[3:]}')
    else:
        raise AssertionError('Please check the device')
    print('Using {}'.format(device))

    isSBERT = False
    if (args.cont and args.mode == 'train') or args.mode != 'train':
        try:
            isSBERT = True if args.name.lower() == 'sbert' else False
            config = Config(config_path)
            if not isSBERT:
                config = Config(config.base_path + '/model/' + args.name + '/' + args.name + '.json')
            else:
                config.dataset_path = {'train': None, 'val': None}
                config.model_path = None
            base_path = config.base_path
        except:
            print('*'*36)
            print('There is no [-n, --name] argument')
            print('*'*36)
            sys.exit()
    else:
        config = Config(config_path)
        base_path = config.base_path
        data_name = config.data

        # make neccessary folders
        os.makedirs(base_path + 'model', exist_ok=True)
        os.makedirs(base_path + 'loss', exist_ok=True)
        os.makedirs(base_path + 'logs', exist_ok=True)
        os.makedirs(base_path + 'data/' + data_name + '/processed', exist_ok=True)

        # define dataset path
        config.dataset_path = make_dataset_path(base_path, data_name, config.train_mode)

        # make model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        config.model_path = model_folder + '/' + config.model_name + '.pt'
        model_json_path = model_folder + '/' + config.model_name + '.json'
        os.makedirs(model_folder, exist_ok=True)
          
        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)
    
    if args.mode == 'benchmark':
        del config.dataset_path['train']
        config.dataset_path['val'] = base_path + 'data/semantic/benchmark/' + args.benchmark + '/data.val'

    trainer = Trainer(config, device, args.mode, args.cont, isSBERT)

    if args.mode == 'train':
        loss_data_path = config.loss_data_path
        print('Start training...\n')
        loss_data = trainer.training()

        print('Saving the loss related data...')
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)

    elif args.mode == 'inference':
        print('Start inferencing...\n')
        trainer.inference('test')
            
    elif args.mode == 'test':
        print('Start testing...\n')
        try:
            trainer.test('test')
        except KeyError:
            trainer.test('val')

    elif args.mode == 'benchmark':
        print('Start testing...\n')
        try:
            trainer.benchmark_test('test')
        except KeyError:
            trainer.benchmark_test('val')

    else:
        print("Please select mode among 'train', 'inference', and 'test'..")
        sys.exit()



if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/')+1] + 'config.json'    

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'inference', 'test', 'benchmark'])
    parser.add_argument('-c', '--cont', type=int, default=0, required=False)
    parser.add_argument('-n', '--name', type=str, required=False)
    parser.add_argument('-b', '--benchmark', type=str, required=False)
    args = parser.parse_args()

    main(path, args)