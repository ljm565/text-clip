import torch
import pickle
import os
import json
import sys
from argparse import ArgumentParser
from train import Trainer
from prepare_chatbot_data import prepare_data_type1, prepare_data_type2, prepare_data_type3, prepare_data_type4
from prepare_sentiment_data import prepare_sdata_type1, prepare_sdata_type2, prepare_sdata_type3, prepare_sdata_type4
from utils.config import Config
from utils.utils_func import collect_all_data, make_vocab_file, make_dataset_path



def main(config_path:Config, args:ArgumentParser):
    device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
    print('Using {}'.format(device))

    if (args.cont and args.mode == 'train') or args.mode != 'train':
        try:
            config = Config(config_path)
            config = Config(config.base_path + '/model/' + args.name + '/' + args.name + '.json')
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

        if data_name == 'koChat':
            # prepare chatbot data
            if not os.path.isdir(base_path + 'data/' + data_name + '/processed/chatbot/'):
                os.makedirs(base_path + 'data/' + data_name + '/processed/chatbot')
                raw_data_path = base_path + 'data/' + data_name + '/raw/chatbot/'
                folder_dict = {'한국어 대화': 'type1',
                                '소상공인 고객 주문 질의-응답 텍스트': 'type2', 
                                '용도별 목적대화 데이터': 'type3', 
                                '주제별 텍스트 일상 대화 데이터': 'type4'}

                for folder_name, type in folder_dict.items():
                    if type == 'type4':
                        prepare_data_type4(raw_data_path, folder_name)
                
                chatbot_data = collect_all_data(base_path + 'data/' + data_name + '/processed/chatbot/')
                all_data_path = base_path + 'data/' + data_name + '/processed/chatbot/all_data.txt'
                special_token_path = base_path + 'data/' + data_name + '/processed/chatbot/special_tokens.txt'
                make_vocab_file(chatbot_data, all_data_path, special_token_path)

            # check tokenizer file
            if not os.path.isdir(base_path + 'data/' + data_name + '/tokenizer/vocab_' + str(config.vocab_size)):
                print('You must make vocab and tokenizer first..')
                sys.exit()
            config.tokenizer_path = base_path+'data/' + data_name + '/tokenizer/vocab_'+str(config.vocab_size)+'/vocab.txt'
        
        else:
            config.dataset_path = make_dataset_path(base_path, data_name)
            if data_name == 'semantic':
                del config.dataset_path['test']
        
        # redefine config
        config.loss_data_path = base_path + 'loss/' + config.loss_data_name + '.pkl'

        # make model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        config.model_path = model_folder + '/' + config.model_name + '.pt'
        model_json_path = model_folder + '/' + config.model_name + '.json'
        os.makedirs(model_folder, exist_ok=True)
          
        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)
    
    trainer = Trainer(config, device, args.mode, args.cont)

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
        trainer.test('test')

    else:
        print("Please select mode among 'train', 'inference', and 'test'..")
        sys.exit()



if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/')+1] + 'config.json'    

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'inference', 'test'])
    parser.add_argument('-c', '--cont', type=int, default=0, required=False)
    parser.add_argument('-n', '--name', type=str, required=False)
    args = parser.parse_args()

    main(path, args)