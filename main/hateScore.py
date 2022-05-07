# -*- coding: utf-8 -*-
'''
@Time    : 2022-01-19 9:17 p.m.
@Author  : datasnail
@File    : hateScore.py
'''
import pandas as pd
import numpy as np

import davidson_model

''' Lexicon is the hate lexicon used in the paper.
    If needed, you can replace `lexicon_dict` with any hate lexicon in the form of a dictionary where the key refers to the word, and value refers to the hate score.
'''
lexicon = pd.read_csv('expandedLexicon.txt', sep = '\t', header = None)
lexicon_dict = {}
max_val = lexicon[1].values.max()
min_val = lexicon[1].values.min()

def getLexDict(x):
    lexicon_dict[x[0].split('_')[0]] = lexicon_dict.get(x[0], (x[1] - min_val)/(max_val - min_val))
    return None

lexicon.apply(lambda x: getLexDict(x), axis = 1)

''' Tokenize function to split the content into words, mostly for the lexicon scores.
    Can be replaced with some better algorithm if needed.
'''
def tokenize(content):
    '''
        Tokenizing a string into an array of words.
        Input: String
        Output: Array of strings
    '''
    return content.split(' ')

''' To get the hate scores for each peice of content in the thread. Model used for now is Davidson et al. with weight 0.6 for incorporating lexicon scores.
    To replace the hate model:-
        Pass in another model to the `hateScore` function.
        Modify line 2 of the `hateScore` function such that the new model returns the confidence score for label `hateful` into `model_score`.
'''
def hateScore(content, model = davidson_model, lexicon = lexicon_dict, weight = 0.6):
    '''
        Get hate score for a peice of content.
        Input:
            content - string
            model - as needed
            lexicon - dict
            weight - float (between 0 and 1)
        Output: float (between 0 and 1)
    '''
    tokens = tokenize(content)
    model_score = model.score_set([content], only_hate = True)[0]
    lexicon_score = np.array([lexicon.get(i,0) for i in tokens]).mean()
    return weight*(model_score) + (1 - weight)*(lexicon_score)

def getHateScoreArr(thread, model = davidson_model, lexicon = lexicon_dict, weight = 0.6):
    '''
        Get the hate scores for the complete thread.
        Input:
            thread - array of strings
            model - as needed
            lexicon - dict
            weight - float (between 0 and 1)
        Output: array of floats (between 0 and 1) [len(Output) = len(thread)]
    '''
    return [hateScore(i, model, lexicon, weight) for i in thread]



thread = ['Fuck whoever made this coronavirus. &amp; who’s idea was it to name a virus after gotdamn beer? 😒']
weight = 0.6
model = davidson_model
lexicon = lexicon_dict
hateScoreArr = getHateScoreArr(thread, model, lexicon, weight)
print(hateScoreArr)
