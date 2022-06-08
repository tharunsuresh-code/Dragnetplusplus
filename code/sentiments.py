# -*- coding: utf-8 -*-
'''
@Time    : 2022-01-19 9:05 p.m.
@Author  : datasnail
@File    : sentiments.py
'''
from torch.nn import AvgPool2d
from scipy import spatial
import torch
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('textattack/xlnet-base-cased-SST-2')
sentiModel = XLNetModel.from_pretrained('textattack/xlnet-base-cased-SST-2')

def cosineSimilarity(x, y):
    '''
        Get cosine similarity between 2 vectors.
        Input:
            x - vector of numbers
            y - vector of numbers
        Output: float (between -1 and 1)
    '''
    return 1-spatial.distance.cosine(x, y)

''' To get the sentiment scores for the thread provided - sentiment similarity between a reply to the original post at index 0.
'''
def sentimentScore(device, content, sentiModel = sentiModel, sentiTokenizer = tokenizer):
    '''
        Get sentiment embeddings from a SOTA DL model. XLNet in this case.
        Input:
            content - string
            sentiModel - as needed (transformers.XLNetModel in this case)
            sentiTokenizer - as needed (transformers.XLNetTokenizer in this case)
        Output: numpy array of numbers representing the sentiment embedding from the sentiModel.
    '''
    inputs = sentiTokenizer(content, return_tensors = "pt").to(device)
    outputs = sentiModel(**inputs)
    last_hidden_states = outputs.last_hidden_state
    m = AvgPool2d((last_hidden_states.shape[1],1), stride = (last_hidden_states.shape[1], 1))
    return (m(last_hidden_states)*last_hidden_states.shape[1]).detach().cpu()

def getSentiScoreArr(device, thread, sentiModel = sentiModel, sentiTokenizer = tokenizer):
    '''
        Get the relative sentiment similarity scores for a thread.
        Input:
            thread - array of strings
            sentiModel - as needed (refer to `sentimentScore`)
            sentiTokenizer - as needed (refer to `sentimentScore`)
        Output: array of numbers between -1 and 1. [len(Output) = len(thread)]
    '''
    sentiEmbeddingsArr = [sentimentScore(device, i, sentiModel, sentiTokenizer) for i in thread]
    return [cosineSimilarity(i.flatten(), sentiEmbeddingsArr[0].flatten()) for i in sentiEmbeddingsArr]


# thread = ['The people of Germany are turning against their leadership as migration is rocking the already tenuous Berlin coalition. Crime in Germany is way up. Big mistake made all over Europe in allowing millions of people in who have so strongly and violently changed their culture','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi']


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# sentiTokenizer = tokenizer
# sentiModel = sentiModel.to(device)
# sentiScoreArr = getSentiScoreArr(device, thread, sentiModel, sentiTokenizer)
# print(sentiScoreArr)
