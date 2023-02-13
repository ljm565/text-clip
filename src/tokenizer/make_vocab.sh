#!/bin/sh

## setup
dpath=../../data/processed/chatbot/all_data.txt
tokens=../../data/processed/chatbot/special_tokens.txt
tpath=../../data/tokenizer
vocab_size=30000

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 vocab_trainer.py --data $dpath --size $vocab_size --output $tpath/vocab_$vocab_size --tokens $tokens