# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:37:38 2020

@author: mrbacco
"""

#### starting with importing the required libraries
import time
import random as rm
from datetime import datetime
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
       "attractive", "bald","beautiful", "chubby", "clean", "dazzling", 
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

try:
    with open('/home/pi/Documents/CODE/PYTHON/ARISTO__PLATO/corpus001.txt', "r") as f4:
        spamreader = csv.reader(f4, delimiter=',')
        test001 = [i for row in spamreader for i in row]
    print("number of words in test001 is ", len(test001))
except FileNotFoundError:
    test001 = []
    print("corpus001.txt not found, continuing without external corpus")


#### start of the main functionality

# Possible sequences of phrases (constant — defined once outside the loop)
trans_name = [["AA","AB","AC","AD"], ["BA","BB","BC","BD"],
              ["CA","CB","CC","CD"], ["DA","DB","DC","DD"]]
# Probabilities matrix (transition matrix)
trans_matrix = [[0.1, 0.3, 0.2, 0.4], [0.2, 0.2, 0.4, 0.2],
                [0.4, 0.3, 0.1, 0.2], [0.2, 0.3, 0.3, 0.2]]

# checking consistency of transition matrix
if sum(trans_matrix[0])+sum(trans_matrix[1])+sum(trans_matrix[2])+sum(trans_matrix[3]) != 4:
    print("\n", "the trans_matrix is not consistent, mrbacco ... ")
else:
    print("\n", "trans_matrix is ok ... ")

# Mapping each state to its word list and row index.
# The four states correspond to transition-matrix rows 0-3 (A-D).
# "adjectives" holds describing words that were labelled "object" in the
# original transition-matrix design; the name is kept consistent with the
# adj list used throughout.
state_words = {"nouns": nouns, "verbs": verbs, "adv": adv, "adjectives": adj}
state_index = {"nouns": 0, "verbs": 1, "adv": 2, "adjectives": 3}
index_state = {0: "nouns", 1: "verbs", 2: "adv", 3: "adjectives"}


# definition of the main model
def sentence_constr(sentences, current_state=None):
    """Build a segment of `sentences` words by walking the Markov chain.

    Parameters
    ----------
    sentences : int
        Number of words to generate in this segment.
    current_state : str or None
        The state to start from.  When None a random state is chosen.

    Returns
    -------
    (list[str], str)
        The generated words and the state reached at the end of the segment,
        so the caller can pass it into the next call to continue the chain.
    """
    if current_state is None:
        current_state = rm.choice(list(state_words.keys()))

    i = 0
    prob = 1
    sentenceList = []

    print("start sentence is: ", current_state)
    while i < sentences:
        idx = state_index[current_state]

        # Pick a word that belongs to the current state
        sentenceList.append(rm.choice(state_words[current_state]))

        # Sample the next state from the transition matrix row for current state
        change = np.random.choice(trans_name[idx], replace=True, p=trans_matrix[idx])
        print("we are at this point in the transition matrix: ", change)

        # Update the running probability and advance the state.
        # The label format is two letters, e.g. "AB": the second letter encodes
        # the destination row (A=0, B=1, C=2, D=3) via its ASCII offset from 'A'.
        next_idx = ord(change[1]) - ord('A')
        prob = prob * trans_matrix[idx][next_idx]
        current_state = index_state[next_idx]

        i += 1
        print("iteration number: ", i)

    return sentenceList, current_state


# main loop
n = 0
while n >= 0:
    now = str(datetime.now().replace(microsecond=0))

    # empty list to save the whole sentence
    list_sentence = []
    current_state = None  # let the chain pick its own starting state

    for iterations in range(1, 8):
        words, current_state = sentence_constr(3, current_state)
        list_sentence.append(words)

    # Flatten all word segments into one readable sentence and print with timestamp
    flat_sentence = " ".join(word for segment in list_sentence for word in segment)
    print(flat_sentence, now)

    time.sleep(2)


