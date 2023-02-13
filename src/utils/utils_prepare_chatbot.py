import re
import os
import random
import pickle


"""
chatbot common utils
"""
def add_punctuation(sentences, list_input=True):
    if list_input:
        if not isinstance(sentences, list):
            raise TypeError
        return [s + '.' if not s[-1] in '.?!' and random.random() < 0.5 else s for s in sentences]    
    
    sentences = sentences + '.' if not sentences[-1] in '.?!' and random.random() < 0.5 else sentences
    return sentences
    
           

"""
chatbot type1 utils
"""
def type1_merge_same_speaker(speaker_ids, sen_ids, sentences):
    assert len(speaker_ids) == len(sen_ids) == len(sentences)

    cur_speaker_id, cur_sen_id = -100, -100
    new_speaker_ids, new_sen_ids, new_sentences = [], [], []

    for speaker_id, sen_id, sentence in zip(speaker_ids, sen_ids, sentences):
        try:
            if cur_speaker_id == speaker_id and 0 < cur_sen_id < sen_id:
                new_sentences[-1] += ' ' + sentence
                continue
        except TypeError:
            break
        
        new_speaker_ids.append(speaker_id)
        new_sen_ids.append(sen_id)
        new_sentences.append(sentence)

        cur_speaker_id, cur_sen_id = speaker_id, sen_id

    assert len(new_speaker_ids) == len(new_sen_ids) == len(new_sentences)
    
    return new_speaker_ids, new_sen_ids, new_sentences



def type1_merge_multiturn(sen_ids, sentences):
    cur_sen_id = -100
    multiturn, multiturns = [], []
    
    for sen_id, sentence in zip(sen_ids, sentences):
        
        if sen_id < cur_sen_id and cur_sen_id != -100:
            multiturns.append(tuple(multiturn))
            multiturn = []
            cur_sen_id = sen_id
            continue

        multiturn.append(sentence)
        cur_sen_id = sen_id
    
    # append the last multi-turn data
    multiturns.append(tuple(multiturn))

    return multiturns



"""
chatbot type3 utils
"""
def type3_merge_same_speaker(lines):
    cur_speaker_id = -100
    new_lines = []

    for s in lines:
        speaker_id, ko = type3_speaker_changer(s[:2])
        if speaker_id == '..':
            new_lines[-1] += ' ' + s
            continue

        assert speaker_id in ['A.', 'B.']
        
        s = s[1:] if ko else s[2:]
        
        if speaker_id == cur_speaker_id and cur_speaker_id != -100:
            new_lines[-1] += ' ' + s
            cur_speaker_id = speaker_id
            continue
        
        new_lines.append(s)
        
    return new_lines



def type3_speaker_changer(speaker_id):
    if speaker_id == '#@':
        return '..', False

    ko_compile = re.compile(r'[ㄱ-ㅣ가-힣0-9]')
    ko = bool(ko_compile.findall(speaker_id))

    speaker_id = re.sub(ko_compile, '.', speaker_id)
    speaker_id = re.sub('[, :;/]', '.', speaker_id)

    return speaker_id, ko



def type3_add_punctuation(sentences):
    if not sentences[-1] in '.?!' and random.random() < 0.5:
        if sentences[-1] == '다':
            return sentences + '.'
        return sentences + random.choice(['.', '?'])
    return sentences



"""
chatbot type4 utils
"""
def type4_merge_same_speaker(lines, file):
    cur_speaker_id = -100
    new_lines = []

    for s in lines:
        # considering which does not have id
        compile = re.findall('([0-9] : )', s)
        
        if len(compile) == 0:
            s = re.sub('[\*]{1,}', '#@이름#', s)
            s = re.sub('[키]{2,}', '', s)
            s = ' '.join(s.split())
            if s == '':
                continue
            try:
                new_lines[-1] += ' ' + s
            except:
                new_lines.append(s)
                cur_speaker_id = '1 : '
            continue
            
        # delete duplicated speaker id
        for pattern in compile[1:]:
            s = s.replace(pattern, '')

        # id sanity check
        speaker_id = s[:4]
        assert re.match('([0-9] : )', speaker_id) != None

        # replace '*' * n system messages
        s = s[4:]
        s = re.sub('[\*]{1,}', '#@이름#', s)

        # delete 키키 and replace duplicated spaces
        s = re.sub('[키]{2,}', '', s)
        s = ' '.join(s.split())

        if s == '':
            continue

        if speaker_id == cur_speaker_id and cur_speaker_id != -100:
            new_lines[-1] += ' ' + s
            cur_speaker_id = speaker_id
            continue
        
        new_lines.append(s)
        
    return new_lines