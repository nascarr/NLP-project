
# coding: utf-8

# In[6]:


import re
import os
import sys
import csv
import time
import json
import collections
from collections import Counter

import numpy as np
import pandas as pd
import random
#import sklearn
from sklearn.model_selection import train_test_split


# In[7]:


def _clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# In[8]:


def clean_data(data, min_utt_number, max_utt_length):
    """
    Clean data: delete duplicates, short threads, long threads, unevaluated threads
    :param data: Pandas DataFrame
    :return data 
    """
    def thread_length(thread):
        lengths = []
        lengths = list(Counter([line['userId'] for line in thread]).values())
        if len(lengths) <= 1:
            min_length = 0
        else:
            min_length = min(lengths)
        return min_length

    def long_utt(thread, max_utt_length):
        result = False
        long_utts = []
        for line in thread:
            if len(line['text']) >  max_utt_length: 
                result = True
                long_utts.append(line['text'])
        return result, long_utts
    
    def if_evaluated(evaluation):
        if_evaluated = False
        for ev in evaluation:
            if ev['quality'] != 0:
                if_evaluated = True
        return if_evaluated
    
    def thread_type(users):
        user_types = []
        for u in users:
            user_type = u['userType']
            if user_type[-4:] == 'Chat':
                user_types.append('Human')
            if user_type[-3:] == 'Bot':
                user_types.append('Bot')
        if user_types == ['Bot', 'Human'] or user_types == ['Human', 'Bot']: 
            return 'human-bot'
        elif user_types == ['Human', 'Human']:
            return 'human-human'
        else:
            print("Wrong user type")
    
    def user_type(u):
        output = None
        user_type = u['userType']
        if user_type[-4:] == 'Chat':
            output = 'Human'
        if user_type[-3:] == 'Bot':
            output = 'Bot'
        return output

    def concatenate_thread_w_users(thread, user_id_dict):
        thread_str = ''
        warning = False
        for th in thread:
            if th['userId'] == user_id_dict['USER0']:
                thread_str += 'USER0 ' + th['text'] + ' EOU '
            elif th['userId'] == user_id_dict['USER1']:
                thread_str += 'USER1 ' + th['text'] + ' EOU '
            else:
                print('Warning: userId in thread isnt equal to any userId in user_id_dict')
                warning = True
            #thread_str = thread_str[:-4]
        if warning == True:
            thread_str = ''
        #print(thread_str)
        return _clean_str(thread_str)

    def get_thread_and_ev(d):
        marks = []
        threads = []
        users = d['users']
        evs = d['evaluation']

        if d['thread_type'] == 'human-bot':
            if user_type(users[0]) == 'Bot':
                bot_id = users[0]['id']
                human_id = users[1]['id']
            else:
                bot_id = users[1]['id']
                human_id = users[0]['id']
            for ev in evs:
                mark = ev['quality']
                if mark != 0:
                    marks.append(mark)
            user_id_dict = {"USER0":human_id, "USER1":bot_id}
            thread = concatenate_thread_w_users(d['thread'], user_id_dict)
            threads.append(thread)

        elif d['thread_type'] == 'human-human':
            marks.append(evs[0]['quality'])
            marks.append(evs[1]['quality'])
            human0_id = evs[0]['userId']
            human1_id = evs[1]['userId']
            threads.append( concatenate_thread_w_users(d['thread'], {"USER0":human0_id, "USER1":human1_id}))
            threads.append( concatenate_thread_w_users(d['thread'], {"USER0":human1_id, "USER1":human0_id}))
        else:
            print("Error: unknown thread type")
        return marks, threads
    
    def thread_params(thread):
        output = None, None
        max_utt_length = 0
        utt_number = len(thread)
        for line in thread:
            max_utt_length = max(max_utt_length, len(line['text'].split()))
        return utt_number, max_utt_length

    data_mod1 = data.drop_duplicates(subset="dialogId")
    data_mod2 = data_mod1[data_mod1['thread'].apply(thread_length) >= min_utt_number]
    data_mod3 = data_mod2[~data_mod2['thread'].apply(lambda x: long_utt(x, max_utt_length)[0])]
    data_mod4 = data_mod3[data_mod3['evaluation'].apply(lambda x: if_evaluated(x))]
    
    print("Number of threads in file", len(data))
    print("Number of deleted duplicates: ", len(data)  - len(data_mod1))
    print("Number of deleted short threads: ", len(data_mod1)  - len(data_mod2))
    print("Number of deleted threads with long utts: ", len(data_mod2) - len(data_mod3))
    print("Number of deleted threads without evaluation: ", len(data_mod3) - len(data_mod4))
    print("Number of remaining threads", len(data_mod4))
    
    data_mod4['thread_type'] = data_mod4['users'].apply(thread_type)
    data_mod5 = data_mod4.copy()
    
    #transform thread, extract mark (=evaluation['quality]) and use human-human threads two times with different marks
    for i,d in data_mod5.iterrows():
        marks, threads = get_thread_and_ev(d)
        data_mod5.at[i, 'evaluation'] = marks[0]
        data_mod5.at[i, 'thread']  = threads[0]
        if len(marks) > 1:
            new_d = (data_mod5.loc[i]).copy()
            new_d['evaluation'] = marks[1]
            new_d['thread'] = threads[1]
            data_mod5 = data_mod5.append(new_d)
    
    
    data_mod6 = data_mod5[['dialogId', 'thread', 'evaluation', 'thread_type']]
    data_mod6.to_csv("data/ev_thread_HRE.csv", index=False, header=['id', 'label', 'content'], columns=['dialogId', 'evaluation', 'thread'])
    
    
    return data_mod6


# In[29]:


def data_to_emb(data_file_path, emb_file_path):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param emb_file_path: word embeddings file path 
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths_th, lengths_utt
    """
    start = time.time()


    emb_f = open(emb_file_path)
    glove_embs = emb_f.read()
    emb_f.close()
    glove_embs = glove_embs.split("\n")[:-1]
    words = []
    vectors = []
    for em in glove_embs:
        values = em.split(' ')
        word = values[0]
        words.append(word)
        vector = np.asarray(values[1:], dtype="float32")
        vectors.append(vector)
    
    #load and transform dataset
    f_data = open(data_file_path)
    data_reader = csv.reader(f_data)
    threads = []
    labels = []
    ids = []
    for row in data_reader:
        thread = row[2].split()
        threads.append(thread)
        labels.append(row[1])
        ids.append(row[0])
    f_data.close()
    threads = threads[1:]
    labels = np.array(labels[1:])
    ids = np.array(ids[1:])
    #lengths = np.array(list(map(len, threads)))
    
    all_threads = [item for sublist in threads for item in sublist]
    vocab_dict = Counter(all_threads)
    vocab = list(vocab_dict.keys())
    
    vocab_embs = np.zeros([100, len(vocab)])
    for word_idx, word in enumerate(words):
        if word in vocab:
            vocab_embs[:, vocab.index(word)] = vectors[word_idx]
    
    #vocab_embs[0, vocab.index('user0')] = 1
    #vocab_embs[1, vocab.index('user1')] = 1
            
    #max_length = max(lengths)
    
    max_utt_number = 101
    max_utt_length = 48
    data = np.zeros([len(threads), max_utt_number, max_utt_length])
    lengths_th = np.zeros([len(threads)])
    lengths_utt = np.zeros([len(threads), max_utt_number])
    users = np.zeros([len(threads), max_utt_number, 2])
    for th_idx, thread in enumerate(threads):
        w_idx = 0
        utt_idx = 0
        #print("th_idx", th_idx)
        for _, word in enumerate(thread):
            if word == "user0":
                users[th_idx, utt_idx, :] = [1, 0]
            elif word == 'user1':
                users[th_idx, utt_idx, :] = [0, 1]

            elif word != 'eou':
                data[th_idx, utt_idx, w_idx] = vocab.index(word)
                w_idx += 1
       
            else: 
                lengths_utt[th_idx, utt_idx] = w_idx
                w_idx = 0
                utt_idx += 1
        lengths_th[th_idx] = utt_idx
        
    data_size = len(data)


    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))

    return ids, users, data, labels, lengths_th, lengths_utt, vocab_embs


# In[30]:


def batch_iter(users, data, labels, lengths_th, lengths_utt, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    #assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size
            userdata = users[start_index: end_index]
            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            seq_lengths_th = lengths_th[start_index: end_index]
            seq_lengths_utt = lengths_utt[start_index: end_index]

            yield userdata, xdata, ydata, seq_lengths_th, seq_lengths_utt




def my_train_test_split(ids, users, data, labels, lengths_th, lengths_utt, random_state=None, test_size=0.1):
    valid_i = []
    train_i = []
    id_dict = Counter(ids)
    unique_ids = list(id_dict.keys())
    unique_ids_train, unique_ids_valid = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    for i, idx in enumerate(ids):
        if np.isin(unique_ids_valid, idx).any():
            valid_i.append(i)
        else:
            train_i.append(i)
    
    users_train = users[train_i]
    users_valid = users[valid_i]
    data_train = data[train_i]
    data_valid = data[valid_i]
    labels_train = labels[train_i]
    labels_valid = labels[valid_i]
    lengths_th_train = lengths_th[train_i]
    lengths_th_valid = lengths_th[valid_i]
    lengths_utt_train = lengths_utt[train_i]
    lengths_utt_valid = lengths_utt[valid_i]
    
    return users_train, users_valid, data_train, data_valid, labels_train, labels_valid,lengths_th_train,lengths_th_valid, lengths_utt_train, lengths_utt_valid     






