
"""
MARKOV_CHAIN_TOOL project

mrbacco04@gmail.com
Q2, 2026

"""

import time
import random as rm
from datetime import datetime
import pandas as pd
import numpy as np
import csv

# Define the corpora
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

# Load the corpus
with open('/home/pi/Documents/CODE/PYTHON/ARISTO__PLATO/corpus001.txt', "r") as f4:
    spamreader = csv.reader(f4, delimiter=',')
    test001=[i for row in spamreader for i in row]

# Create a Markov chain model
class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.transition_matrix = np.zeros((len(nouns), len(verbs)))
        self.object_matrix = np.zeros((len(adj), len(adv)))

    def build_model(self, sentences):
        for i in range(len(sentences) - self.order):
            sentence = tuple(sentences[i:i + self.order])
            next_sentence = tuple(sentences[i + self.order:i + self.order + 1])
            if next_sentence not in [tuple(x) for x in list(self.transition_matrix)]:
                # Increment the transition count
                current_index = np.where(tuple(x) == sentence)[0][0]
                next_index = np.where(tuple(x) == next_sentence)[0][0]
                self.transition_matrix[current_index, next_index] += 1

    def get_next_word(self, word):
        if word in nouns:
            return rm.choices(adj, weights=self.object_matrix[nouns.index(word)])
        elif word in verbs:
            return rm.choices(nouns, weights=self.transition_matrix[verbs.index(word)])
        elif word in adj:
            return rm.choices(adv, weights=self.object_matrix[adj.index(word)])
        else:
            return rm.choice(list(nouns + verbs + adj))

    def generate_sentence(self):
        sentence = [rm.choice(nouns)]
        for _ in range(self.order - 1):
            next_word = self.get_next_word(sentence[-1])
            sentence.append(next_word)
        return ' '.join(sentence)

# Create a Markov chain model instance
model = MarkovChain()

# Build the model
model.build_model(test001)

# Generate sentences
for _ in range(10):
    print(model.generate_sentence())
