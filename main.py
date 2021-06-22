# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:29:36 2020
updated June 2021
@author: mrbacco 2021
"""

import numpy as np
import random as rm
import time
from datetime import datetime

# The sentencepace
sentence = ["Subject","Verb","Object"]

# Possible sequences of events
transitionName = [["SS","SV","SO"],["VS","VV","VO"],["OS","OV","OO"]]

# Probabilities matrix (transition matrix)
transitionMatrix = [[0.2,0.3,0.5],[0.1,0.6,0.3],[0.2,0.7,0.1]]

# checking consistency of transition matrix
if sum(transitionMatrix[0])+sum(transitionMatrix[1])+sum(transitionMatrix[1]) != 3:
    print("the matrix is not consistent, mrbacco ... ")
else: print("All is gonna be okay mrbacco ... ")

x = 3
while x >=0:
    num1 = rm.randrange(0,3) # picking a number randomly from 1 to 3
    # definition of the main model
    def sentence_forecast(act):
        sentenceToday = sentence[num1] # Choose the starting state, randomly
        sentenceList = [sentenceToday]
        print("start sentence is: ", sentenceToday)
        i = 0
        prob = 1
        while i != act:
            if sentenceToday == "Subject":
                change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
                if change == "SS":
                    prob = prob * 0.2
                    sentenceList.append("Subject")
                    pass
                elif change == "SO":
                    prob = prob * 0.5
                    sentenceToday = "Object"
                    sentenceList.append("Object")
                else:
                    prob = prob * 0.3
                    sentenceToday = "Verb"
                    sentenceList.append("Verb")
            elif sentenceToday == "Object":
                change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
                if change == "VO":
                    prob = prob * 0.5
                    sentenceList.append("Object")
                    pass
                elif change == "VS":
                    prob = prob * 0.3
                    sentenceToday = "Subject"
                    sentenceList.append("Subject")
                else:
                    prob = prob * 0.2
                    sentenceToday = "Verb"
                    sentenceList.append("Verb")
            elif sentenceToday == "Verb":
                change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
                if change == "OV":
                    prob = prob * 0.2
                    sentenceList.append("Verb")
                    pass
                elif change == "OS":
                    prob = prob * 0.3
                    sentenceToday = "Subject"
                    sentenceList.append("Subject")
                else:
                    prob = prob * 0.5
                    sentenceToday = "Object"
                    sentenceList.append("Object")
            i += 1
        return sentenceList
        
    # To save every sentenceList
    list_sentence = []
    count = 0

    # `Range` starts from the first count up until but excluding the last count
    for iterations in range(1,11000):
        list_sentence.append(sentence_forecast(2))

    # Check out all the `sentenceList` we collected    
    print("\n",list_sentence)

    # Iterate through the list to get a count of all activities ending with initial state
    for smaller_list in list_sentence:
        if(smaller_list[2] == sentence[num1]):
            count += 1
            # print("\n", sentence[num1])
            # print(count)

    # Calculate the probability of starting from state:'Subject' and ending at state:'Object'
    percentage = (count/10000) * 100
    print("\n", "The probability of starting at: ", sentence[num1]," and ending at: ", sentence[num1], " = " + str(percentage) + "%")
    now = str(datetime.now().replace(microsecond=0))
    print("time now is: ", now)
    time.sleep(10)
