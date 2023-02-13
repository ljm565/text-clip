from tokenizers import BertWordPieceTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,required=True)
parser.add_argument("--tokens", default='nothing', type=str,required=False)
parser.add_argument("--size", type=int,required=True)
parser.add_argument("--output", type=str,required=True)
args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(lowercase=False)

if args.tokens == 'nothing':
    tokenizer.train(files=args.data, vocab_size=args.size, min_frequency=5)
else:
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
    with open(args.tokens, 'r') as f:
        tokens = f.readlines()
    
    for tok in tokens:
        special_tokens.append(tok.strip())
        
    tokenizer.train(files=args.data, vocab_size=args.size, min_frequency=3, special_tokens=special_tokens)

tokenizer.save_model(args.output) 

