# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:37:38 2020

@author: mrbacco
"""

#### starting with importing the required libraries
import time
import random as rm
from datetime import datetime
import pandas as pd
import numpy as np
import csv

#### definition of the lists to use a s source/corpora
nouns = ["puppy", "car", "rabbit", "girl", "monkey", "mother", "father", "baby",
         "child", "teenager", "grandmother", "student", "teacher", "minister", 
         "businessperson", "salesclerk", "woman", "man", "lion", "tiger", "bear", 
         "dog", "cat", "alligator", "cricket", "bird", "wolf"]

verbs = ["runs", "hits", "jumps", "drives", "barfs", "enjoys", "lives","is", 
         "has", "does", "says", "makes", "knows", "thinks", "takes", "sees", 
         "comes", "wants", "looks", "uses", "finds", "gives", "tells", "yells"]

adj = ["adorable", "clueless", "dirty", "odd", "stupid", "useless", "funny", 
       "", "attractive", "bald","beautiful", "chubby", "clean", "dazzling", 
       "drab", "elegant", "fancy", "fit", "flabby", "glamorous", "gorgeous",
       "handsome", "long", "magnificent", "muscular", "plain", "plump", "quaint", 
       "scruffy", "shapely", "short", "skinny", "stocky", "ugly", "unkempt", 
       "unsightly"]

adv = ["crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally.", 
       "happily.", "boldly", "bravely", "brightly", "cheerfully", "deftly",
       "devotedly", "eagerly", "elegantly", "faithfully", "fortunately", 
       "gleefully", "gracefully", "honestly", "innocently", "justly", "kindly",
       "obediently", "perfectly", "politely", "powerfully", "safely", 
       "victoriously", "warmly", "vivaciously"]

with open('/home/pi/Documents/CODE/PYTHON/ARISTO__PLATO/corpus001.txt', "r") as f4:
    spamreader = csv.reader(f4, delimiter=',')
    test001=[i for row in spamreader for i in row]
print("number of words in test001 is "  , len(test001))


#### start of the main functionality


# main loop
n = 0
while n >= 0:
    now = str(datetime.now().replace(microsecond=0))

    # Possible sequences of phrases
    trans_name = [["AA","AB","AC", "AD"],["BA","BB","BC", "BD"],["CA","CB","CC", "CD"],["DA","DB","DC", "DD"]]
    # Probabilities matrix (transition matrix)
    trans_matrix = [[0.1, 0.3, 0.2, 0.4], [0.2, 0.2, 0.4, 0.2], [0.4, 0.3, 0.1, 0.2], [0.2, 0.3, 0.3, 0.2]]

    # checking consistency of transition matrix
    if sum(trans_matrix[0])+sum(trans_matrix[1])+sum(trans_matrix[2])+sum(trans_matrix[3]) != 4:
        print("\n","the trans_matrix is not consistent, mrbacco ... ")
    else: print("\n", "trans_matrix is ok ... ")

    # definition of the main model
    def sentence_constr(sentences):
        i = 0
        prob = 1
        while i != sentences:
            sentence1 = ["nouns", "verbs", "adv", "object"]
            start_sentence = rm.choice(sentence1)  # Choose the starting sentence, randomly
            sentenceList = []
            print("start sentence is: ", start_sentence)
            if start_sentence == "nouns":
                change = np.random.choice(trans_name[0],replace=True,p=trans_matrix[0])
                if change == "AB":
                    prob = prob * 0.3
                    sentenceList.append(rm.choice(nouns))
                    print("we are at this point in the transition matrix: AB")
                    pass
                if change == "AC":
                    prob = prob * 0.2
                    start_sentence = "verbs"
                    sentenceList.append(rm.choice(verbs))
                    print("we are at this point in the transition matrix: AC")
                elif change == "AD":
                    prob = prob * 0.4
                    start_sentence = "adv"
                    sentenceList.append(rm.choice(adj))
                    print("we are at this point in the transition matrix: AD")
                else:
                    prob = prob * 0.1
                    start_sentence = "object"
                    sentenceList.append(rm.choice(adv))
                    print("we are at this point in the transition matrix: AA")
            elif start_sentence == "verbs":
                change = np.random.choice(trans_name[1],replace=True,p=trans_matrix[1])
                if change == "BA":
                    prob = prob * 0.2
                    sentenceList.append(rm.choice(verbs))
                    print("we are at this point in the transition matrix: BA")
                    pass
                if change == "BB":
                    prob = prob * 0.2
                    start_sentence = "nouns"
                    sentenceList.append(rm.choice(nouns))
                    print("we are at this point in the transition matrix: BB")
                elif change == "BC":
                    prob = prob * 0.2
                    start_sentence = "adv"
                    sentenceList.append(rm.choice(adj))
                    print("we are at this point in the transition matrix: BC")
                else:
                    prob = prob * 0.4
                    start_sentence = "object"
                    sentenceList.append(rm.choice(adv))
                    print("we are at this point in the transition matrix: BD")
            elif start_sentence == "adv":
                change = np.random.choice(trans_name[2],replace=True,p=trans_matrix[2])
                if change == "CA":
                    prob = prob * 0.4
                    sentenceList.append(rm.choice(adv))
                    print("we are at this point in the transition matrix: CA")
                    pass
                if change == "CB":
                    prob = prob * 0.3
                    start_sentence = "verbs"
                    sentenceList.append(rm.choice(nouns))
                    print("we are at this point in the transition matrix: CB")
                elif change == "CC":
                    prob = prob * 0.1
                    start_sentence = "adv"
                    sentenceList.append(rm.choice(verbs))
                    print("we are at this point in the transition matrix: CC")
                else:
                    prob = prob * 0.2
                    start_sentence = "object"
                    sentenceList.append(rm.choice(adj))
                    print("we are at this point in the transition matrix: CD")
            elif start_sentence == "object":
                change = np.random.choice(trans_name[3],replace=True,p=trans_matrix[3])
                if change == "DA":
                    prob = prob * 0.2
                    sentenceList.append(rm.choice(adj))
                    print("we are at this point in the transition matrix: DA")
                    pass
                if change == "DB":
                    prob = prob * 0.2
                    start_sentence = "verbs"
                    sentenceList.append(rm.choice(nouns))
                    print("we are at this point in the transition matrix: DB")
                elif change == "DC":
                    prob = prob * 0.3
                    start_sentence = "adv"
                    sentenceList.append(rm.choice(verbs))
                    print("we are at this point in the transition matrix: DC")
                else:
                    prob = prob * 0.4
                    start_sentence = "nouns"
                    sentenceList.append(rm.choice(adv))
                    print("we are at this point in the transition matrix: DD")
            i += 1
            print ("iteration number: ", +i)
        return sentenceList

    # empty list to save the whole sentence
    list_sentence = []

    for iterations in range(1,8):
        list_sentence.append(sentence_constr(3)) 
    
    print(*list_sentence, now)

    time.sleep(2)


