#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:37:29 2022

@author: lizziehe
"""

import numpy as np

## Test data preprocessing -句子
import nltk
#nltk.download('punkt')

def parser(file):
    lines = open(file).readlines()
    original_list=[]
    corrected_list=[]
    for line in lines:
        #print(line)
        #indexes=[]
        tokens = nltk.word_tokenize(line) # Tokenize each sentence
        #print(tokens)
        original = tokens
        corrected = tokens
        #Split the data by the '|' separator to get the Original and Corrected Sentences
        for token in tokens:
            if("|" in token):
                index_ind = tokens.index(token)
                word_orig = tokens[tokens.index(token)].split('|')[0]
                word_corrected = tokens[tokens.index(token)].split('|')[1].replace("_"," ")
                original = [word_orig if word==tokens[tokens.index(token)] else word for word in original]
                corrected = [word_corrected if word==tokens[tokens.index(token)] else word for word in corrected]
                #indexes.append(index_ind)
        
        original_list.append(original)
        corrected_list.append(corrected)
    return original_list,corrected_list

original_list,corrected_list = parser("holbrook.txt")




## Build language model - bigram model; estimate the probability of a word, P(w)

#nltk.download('reuters')
from nltk.corpus import reuters
corpus = reuters.sents()

term_count = {}
bigram_count = {}
for doc in corpus:
    doc = ['<s>'] + doc   # '<s>'表示开头
    for i in range(0, len(doc)-1):
        # bigram: [i,i+1]
        term = doc[i]          # term是doc中第i个单词
        bigram = doc[i:i+2]    # bigram为第i,i+1个单词组成的
        
        if term in term_count:
            term_count[term] += 1   # 如果term存在term_count中，则加1
        else:
            term_count[term] = 1    # 如果不存在，则添加，置为1
        
        bigram = ' '.join(bigram)
        if bigram in bigram_count:
            bigram_count[bigram] += 1
        else:
            bigram_count[bigram] = 1
            
            
## Build chanel model - edit distance; P(c|w)
# Build a function that calculates all words with minimal edit distance to the misspelled word. Steps are as follows:
# 1. Collect the set of all unique tokens in data
# 2. Find the minimal edit distance, that is the lowest value for the function edit_distance between token and a word in data
# 3. Output all unique words in data that have this same (minimal) edit_distance value

from nltk.metrics.distance import edit_distance
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_candidates(token):
    word_corrected_list = [item for sublist in corrected_list for item in sublist]
    # Get the unique list of words
    word_corrected_list_unique = unique(word_corrected_list) 
    # Get the distance w.r.t the first token
    dist =edit_distance(token,word_corrected_list_unique[0])
    
    min_dist_word =[]
    for word in word_corrected_list_unique:
        # Compare the distance with the subsequent tokens and update the list accordingly
        if(edit_distance(token,word) < dist):
            dist = edit_distance(token,word)
            min_dist_word.clear()
            min_dist_word.append(word)
        elif(edit_distance(token,word) == dist):
            min_dist_word.append(word)
    return min_dist_word
#print(get_candidates("thee"))



## Correction function 
from tqdm import tqdm

pred_list=[]
for line in tqdm(original_list[:100]):
    
    V = len(term_count.keys())
    j = 0
    sentence_corrected = []
    
    for word in line:
        if word in term_count.keys():
            sentence_corrected.append(word)
        else:
            #Step1: 生成所有的(valid)候选集合
            candidates = get_candidates(word)
        
            probs=[]
            # 对于每一个candidate, 计算它的score
            # score = p(correct)*p(mistake|correct)
            #       = log p(correct) + log p(mistake|correct)
            # 返回score最大的candidate
            for candi in candidates:
                prob = 0
                #计算语言模型的概率
                forward_word= line[j-1]+" "+candi      #考虑前一个单词,出现like playing的概率
                if forward_word in bigram_count and line[j-1] in term_count.keys():
                    prob += np.log((bigram_count[forward_word]+1.0)/(term_count[line[j-1]]+V))
        
                else:
                    prob += np.log(1.0/V)
        
                if j+1 < len(line):               #考虑后一个单词，出现playing football的概率
                    backward_word= candi + " " + line[j+1]
                    if backward_word in bigram_count and candi in term_count:
                        prob += np.log((bigram_count[backward_word] + 1.0)/(term_count[candi]+V))
                    else:
                        prob += np.log(1.0/V)
                probs.append(prob)
            max_idx = probs.index(max(probs))
            sentence_corrected.append(candidates[max_idx])
            j=j+1
        
    pred_list.append(sentence_corrected)



## Accuracy function

def accuracy(corrected_list,pred_list): 
    total_count = 0
    corrected_count = 0
    a=0
    for line in pred_list:
        word=corrected_list[a]
        index = len(line)
        i = 0
        while(i<index):
            #Compare each word of the Generated Output with the Corrected List
            if(word[i] == line[i]):
                corrected_count += 1
                total_count += 1
            else:
                total_count += 1
            i +=1
        a +=1
    return round(corrected_count/total_count, 5)

# Print the accuracy and the total execution time
print("Accuracy : " + str(accuracy(corrected_list,pred_list)*100) + " percent")

          
            
            
            
            
            
            
            
            