import tensorflow as tf
import json, os

os.environ["TF_KERAS"] = "1"
import re
# from preprocessor import tokenize as tt
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from keras_self_attention import SeqSelfAttention
# import keras
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import TFBertModel, BertConfig
from transformers import DistilBertTokenizer, RobertaTokenizer

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, ' ', parsed_text)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    return parsed_text


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'),
            parsed_text.count('HASHTAGHERE'))


import string
table = str.maketrans('', '', string.punctuation)

def tokenize(tweet):
    stripped = [w.translate(table) for w in tweet.split()]
    tweet = " ".join(stripped).strip()
    #     print(tweet)
    return tweet.split()


label_map_original = {'none': 0, 'racism': 1, 'sexism': 2}
label_map_modified = {
    0: 0,  #NH
    1: 1,  #H
    2: 1  #H
}
label_map_fountana_org = {'abusive': 0, 'normal': 1, 'hateful': 2, 'spam': 3}
hate_map = {0: "NON HATE", 1: "HATE"}
reverse_hate_map = {"NON HATE": 0, "HATE": 1}

def load_data():
    texts = []
    labels = []
    file = os.path.join('data', 'fountana_norm_HAclean.json')
    with open(file, 'r') as f:
        ft_data = json.load(f)
    for each_tweet in ft_data:
        tweet = tokenize(ft_data[each_tweet]['text'])
        texts.append(tweet)
        labels.append(ft_data[each_tweet]['label'])
    return texts, labels

def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence,
                                       add_special_tokens=True,
                                       max_length=128,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])
    input_ids = np.asarray(input_ids, dtype='int32')
    input_masks = np.asarray(input_masks, dtype='int32')
    input_segments = np.asarray(input_segments, dtype='int32')
    return input_ids, input_masks, input_segments

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        output = K.sum(output, axis=1)
        output = K.expand_dims(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()

distil_bert = 'bert-base-uncased'  # Pick any desired pre-trained model
# Defining BERT tokonizer
tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
                                                do_lower_case=True,
                                                add_special_tokens=True,
                                                max_length=128,
                                                pad_to_max_length=True)
config = BertConfig(dropout=0.2, attention_dropout=0.2)
config.output_hidden_states = False
transformer_model = TFBertModel.from_pretrained(distil_bert, config=config)

input_ids_in = tf.keras.layers.Input(shape=(128, ),
                                     name='input_token',
                                     dtype='int32')
input_masks_in = tf.keras.layers.Input(shape=(128, ),
                                       name='masked_token',
                                       dtype='int32')

embedding_layer = transformer_model(input_ids_in,
                                    attention_mask=input_masks_in)[0]

X = tf.keras.layers.SimpleRNN(128)(embedding_layer)
# print(X.shape)
X = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(X)
# print(X.shape)
X = SeqSelfAttention(attention_activation='sigmoid')(X)
X = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(X)

# print(X.shape)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.2)(X)
X = tf.keras.layers.Dense(2, activation='softmax')(X)
model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

for layer in model.layers[:3]:
    layer.trainable = False
# model.summary()

model.compile(optimizer=Adam(lr=0.00001),
              loss='mean_squared_error',
              metrics=['acc'])
model.load_weights('fountana/fountana_weights')

def score_set(test_set, only_hate=False):
    input_ids, input_masks_in, input_segment = tokenize(test_set, tokenizer)
    X_test = [input_ids,input_masks_in]
    y_pred = model.predict(X_test)
    if only_hate:
        return y_pred[:,1]
    else:
        return y_pred

if __name__ == '__main__':
    s = score_set(['Hi, good','fuck'], only_hate=True)
    print(s)